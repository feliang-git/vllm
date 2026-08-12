# Integrating MoonEP as a vLLM MoE dispatcher — feasibility and design

Status: **design + feasibility, no implementation yet.** Two hard constraints
found by reading MoonEP's code scope this work considerably; they are stated
first because they decide what is worth building.

Companion to the MoonEP planner/backend split
(`feliang-git/MoonEP@feat/pluggable-alloc-policy`,
`feliang-git/moe_load_balancer@feat/moonep-policy`).

---

## 1. Blockers found before writing code

### B1. MoonEP is bf16-only. There is no fp8 path at all.

```
moonep/api.py:852   assert w.dtype == torch.bfloat16 and w.is_contiguous()
moonep/api.py:929   assert hidden_nvsh.dtype == torch.bfloat16
moonep/api.py:352   hidden_buf = create_nvl_dist_tensor([NvS_padded, H], torch.bfloat16, ...)
```

A grep for `fp8|float8|scale` across `moonep/*.py` returns **nothing**. The
communication buffers, the expert-weight prefetch, dispatch and combine are all
hard-coded bf16.

This is a real divergence from UltraEP, which explicitly added
`feat(fp8): support low-precision weight format with scale factors` and carries
`fc1_weight_scales` / `fc2_weight_scales` through its transfer API.

**Consequence.** Combined with MoonEP being intra-node only (its NVL shared
buffer uses IPC/multicast within one NVLink domain), the flagship serving
configuration is arithmetically out of reach:

| model | bf16 weights | fits on 8×H20-3e (1128 GB)? |
|---|---|---|
| DeepSeek-R1 / V3 (671B) | ~1342 GB | **no** |
| DeepSeek-V2 (236B) | ~472 GB | yes |
| DeepSeek-V2-Lite (16B) | ~32 GB | yes |
| Qwen3-30B-A3B | ~60 GB | yes |

So MoonEP-in-vLLM cannot serve fp8 DeepSeek-R1/V3, which is the configuration
the LPLB benchmarks in this project use. It *can* serve **DeepSeek-V2**, which
happens to be this project's stated verification scope.

Options: (a) scope to bf16 DeepSeek-V2; (b) add fp8 + scale support to MoonEP
upstream — a substantial change to dispatch, combine, prefetch and grad_reduce.

### B2. Expert weights must be `[E+B]` and NVLink-mapped — this is not a dispatcher-only change

```
moonep/api.py:852   assert w.ndim == 3 and w.shape[0] == ctx['E'] + ctx['B']
```

`prefetch_weight` takes `full_gate/up/down_weight` covering **all E global
experts plus B prefetch slots**, where the remote rows are NVLink-mapped views
of other ranks' memory (`create_nvl_single_owner_tensor`, see
`benchmarks/bench_prefetch.py:106`).

vLLM's `FusedMoE` allocates only the rank's **local** experts, as ordinary
tensors. MoonEP therefore requires changes to **weight allocation and loading**,
not just to a `FusedMoEPrepareAndFinalize` subclass:

- expert weight tensors allocated with `B` extra slots
- allocated inside MoonEP's NVL-distributed buffers so peers can read them
- the expert GEMM must index `E+B` rows and consume MoonEP's `cu_seqlens`
  grouping, which has `E+B` groups — the trailing `B` being *duplicates* of hot
  remote experts, not distinct experts

This mirrors why the SGLang↔UltraEP integration touched `model_runner.py`
(+117 lines) and added `expert_placement_state.py`, rather than living purely in
the dispatcher.

### B3. Static `S`, and a training-shaped API

`Buffer(S, H, K, E, R, ...)` fixes tokens-per-rank at construction, so vLLM must
pad every batch to a fixed maximum. Prefill at the chunked-prefill cap is a good
fit; **decode (S≈1) pays the full padding cost and should be expected to
regress.** Separately, `reduce_grad` and the backward paths are dead code under
inference — roughly half of MoonEP's test suite does not cover the vLLM path.

---

## 2. The integration seams in vLLM

Confirmed against `feliang-git/vllm@12bea3e`. DeepEP-v2 is the closest sibling
and the right model to copy.

| concern | file | note |
|---|---|---|
| dispatcher | `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py:23` | `DeepEPV2PrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular)` |
| abstract contract | `vllm/model_executor/layers/fused_moe/modular_kernel.py` | must implement `prepare` / `finalize`; `prepare_async` / `finalize_async` optional |
| comm manager | `vllm/distributed/device_communicators/all2all.py:1005` | `DeepEPV2All2AllManager(All2AllManagerBase)` |
| selection | `vllm/model_executor/layers/fused_moe/all2all_utils.py:50,105,205` | `use_deepep_v2_kernels` flag path |
| EPLB | `vllm/distributed/eplb/` | `eplb_state.py`, `rebalance_execute.py`, `policy/abstract.py` |

### Mapping `prepare()` onto MoonEP

vLLM calls:

```python
prepare(a1, topk_weights, topk_ids, num_experts, expert_map,
        apply_router_weight_on_input, quant_config, defer_input_quant)
```

MoonEP wants:

```python
buffer.dispatch(hidden_sh, route_weights_sk, topk_experts_sk, tokens_per_expert)
```

Gaps:
- **`tokens_per_expert` does not exist in vLLM's call.** It must be produced by a
  `bincount` over `topk_ids` (`[E]` int32, exact) before dispatch — an extra
  kernel on the critical path that DeepEP does not need.
- **`expert_map` is meaningless to MoonEP.** MoonEP assumes `home(e) = e // epn`
  globally; it does not consume a global→local expert map. Any non-identity
  `expert_map` (which is exactly what vLLM's EPLB produces) is either unsupported
  or must be folded into MoonEP's expert numbering. **This is the concrete
  friction point for composing MoonEP with vLLM's EPLB** — see §3.
- **Output grouping is `E+B`, not `E`.** `cu_seqlens` covers `E+B` padded groups.
  `ExpertTokensMetadata` must be derived from it, and the experts layer must
  understand that the trailing `B` groups are duplicate hot experts whose outputs
  combine back to their home expert.

---

## 3. Where MoonEP and vLLM's EPLB meet

Measurement on the MLB branch (`scripts/moonep_policy_headroom.py`) showed the
allocation step has **no headroom** — MoonEP's greedy attains perfect rank
balance in 38/38 cases. The headroom is in *placement*: a balanced assignment of
expert **homes** removes **62%** of migrated tokens, and the split is by failure
mode, not degree:

- skew spread over many experts → 85–100% of migration removed
- skew concentrated on one expert → **0%**, no home assignment can split it
- `epn == 1` → 0%, the partition is forced up to permutation

So EPLB placement and MoonEP migration fix **disjoint** failure modes, which is
the argument for running both. The mechanical obstacle is the `expert_map` gap
above: vLLM's EPLB expresses placement as a global→local expert map, and MoonEP
hard-codes `home(e) = e // epn`. Composing them means teaching MoonEP a home map
— a modest change to planning step 1/2, but an upstream one.

**Caveat:** the 62% comes from synthetic adversarial distributions, not real
routing traces. Re-measure on DeepSeek traces before building on it.

---

## 4. Proposed scope

Ordered so each step is independently verifiable and the risky part comes last.

1. **Scope to bf16 DeepSeek-V2**, matching this project's stated verification
   scope and sidestepping B1. Explicitly out of scope: fp8 R1/V3.
2. `MoonEPAll2AllManager` + `--moe-a2a-backend moonep` plumbing. Small, testable
   in isolation.
3. Weight-layout work (B2): allocate expert weights `[E+B]` in MoonEP's NVL
   buffers during weight loading. **The real cost of this PR.**
4. `MoonEPPrepareAndFinalize`, forward-only, raising clearly on any backward
   entry point. Pad `S` to the chunked-prefill cap.
5. Correctness gate: greedy-decode output must match the DeepEP-v2 backend
   exactly on a fixed prompt set.
6. Perf gate: prefill throughput + TTFT vs DeepEP-v2, 3-rep counterbalanced.
   **Report decode separately and expect a regression** from `S` padding.
7. Only then consider the EPLB composition in §3.

## 5. Recommendation

Steps 1–2 are cheap. **Step 3 is the decision point**: it changes vLLM's weight
loading for a backend that cannot serve the fp8 models this project benchmarks.
Before committing to it, decide whether the target is a *demonstration* on
DeepSeek-V2 bf16 or a *production* path — the latter requires fp8 support in
MoonEP first, which is upstream work of comparable size to everything done so
far.
