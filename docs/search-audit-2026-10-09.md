# HC/mHC coverage audit — 2026-10-09

This ledger records what changed from the 2026-08-25 README snapshot and what remains unresolved. It is a reproducible gap audit, not proof that every public HC artifact has been found. Dates in the publication timeline use UTC and refer to the first verifiable relevant version, not the latest revision.

## Result

| Index | Before | After |
|---|---:|---:|
| Counted publications | 62 | 71 |
| Direct papers / formal manuscripts | 41 | 49 |
| Adjacent/adoption / formal comparisons | 21 | 22 |
| Indexed but not counted records | 3 | 3 |
| Training/research repository roots | 92 | 104 |
| Inference/serving repository roots | 39 | 42 |
| Total repository roots | 131 | 146 |

One of the 146 repository rows is historical and currently unavailable. All 145 accessible roots have refreshed GitHub Stars and latest default-branch commit metadata captured on **2026-10-09 03:18–03:19 UTC**. Commit ordering uses committer timestamps, not `updated_at` or `pushed_at`; Stars measure the whole repository, not its HC path.

## Added formal records

| First relevant public date | Record | Classification and verification |
|---|---|---|
| 2026-10-08 | [Multi-Head mHC: No Gain at Equal Parameters](https://github.com/marcoshernanz/multi-head-mhc/blob/main/paper/main.tex) | Direct analysis. Named author Marcos Hernanz; complete LaTeX source, parameter/LR controls, and 626 reported raw training runs. The first paper-source commit is [f9b3e64](https://github.com/marcoshernanz/multi-head-mhc/commit/f9b3e6475e4862a458e466d0cb64dc8b9d2141ab), timestamped 21:07:49 UTC; the repository was created at 21:08:02 UTC on the same date. AI-assisted provenance is disclosed; no compiled full PDF, arXiv/DOI, or independent review was located. |
| 2026-10-05 | [The Birkhoff Geometry of Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2610.06653) | Direct analysis; first arXiv submission 16:33:15 UTC. Mean/difference transport channels, singular-value memory horizon, vertex viscosity, and truncated Sinkhorn; proofs and numerical experiments. The mean-preservation statement concerns the residual transport operator, not the entire nonlinear layer. |
| 2026-09-27 | [Residual-Stream Burden Shapes Representation Learning in Diffusion Transformers](https://arxiv.org/abs/2609.33895) | Direct application/architecture; first submission 20:17:39 UTC, revised October 6. SiHC mechanism, mHC interventions, author-linked code, training, and fused kernels were inspected. |
| 2026-09-26 | [DepthBench](https://arxiv.org/abs/2609.32534) | Adjacent/comparison; first submission 12:15:42 UTC. Includes actual HC/mHC training comparisons and residual-product diagnostics, not merely citations. |
| 2026-09-10 | [DeepSeek-V4.1-Flash technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) | Direct method/systems despite the broader model-report scope: §2.4.1 (printed pp. 12–13) introduces Single-Pass mHC and Mega-mHC. The [official announcement](https://www.deepseek.com/en/news/deepseek-v4-1-flash/) and first PDF [upload 97fcff4](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/commit/97fcff464d0daf011178cd857397845aaf5f88fb) agree on September 10 (05:33:07 UTC upload). The reported traffic reduction is deployment-kernel activation I/O, not a claim of halving all training activations, parameters, FLOPs, or end-to-end runtime. |
| 2026-09-04 | [How Does mHC Use Its Residual Streams?](https://arxiv.org/abs/2609.05309) | Direct analysis; first submission 16:02:20 UTC. DeepSeek-V4-Flash routing and residual-mixer interventions. Empirical conclusions remain specific to that model and evaluation. |
| 2026-09-02 | [oHC](https://arxiv.org/abs/2609.02672) | Direct method; first submission 14:45:42 UTC. Orthogonal mixing and quaternion SO(4) parameterization. Author project currently contains the paper/assets, not released implementation code. |
| 2026-08-31 | [Qwen3.8-Next architecture report](https://arxiv.org/abs/2608.30320) | Direct architecture; first submission 06:35:07 UTC. GatedResidual is an HC-derived connection rule with material mechanism/ablation evidence. The model's rank-320 read controller is not compressed latent residual-state storage. |
| 2026-08-31 | [Uni-mHC](https://zenodo.org/records/22183961) | Direct independent manuscript; Zenodo Published/Created August 31, modified September 5. Stable named pen-name authorship, PDF, code, and logs; small-scale evidence, not independently reviewed. Its derived matrix `S = abs(U)^2` is doubly stochastic; the applied phase-projected real mixer `W` is signed and need not be doubly stochastic or norm-isometric. |

## Added repositories and code boundary

The canonical README tables contain the captured commit, Star count, category, and inspected source path for each new root.

Training/research additions are **6zHAOyi/s2HC**, **Yutong-Feng/FlowNet**, **cactus-compute/needle**, **DDDMUC/Uni-mHC**, **alint77/flash-mHC**, **wdlctc/hyper-connection-factory**, **Dojo2024/residual-streams**, **marcoshernanz/multi-head-mhc**, **tongtongliang/residual-stream-burden**, **poisson-labs/mhc-repro**, and the paper/assets-only **Meiyim/oHC-Orthogonal-Hyper-Connections-on-SO4-via-Quaternions** project.

Inference/serving additions are **gty111/gLLM**, **zouyee/dmlx**, **stew675/llama-cpp-rdna-boosts**, and **sybil-solutions/qwen38-flash-next-b70-offload**. The previously indexed **tile-ai/tilelang-musa** root is moved to training/research because its current examples include a community-written backward reference; hence the net block-count changes are +12 and +3, despite adding eleven and four new roots respectively.

Important limitations:

- Dojo2024 implements HC/Frac, while mHC/mHC-lite are still planned.
- Poisson's router pools across sequence positions and shares routing within a sequence; it is not a faithful token-conditioned causal mHC reproduction.
- Uni-mHC's default small-scale experiments use static mixing; its optional mean-conditioned topology is not the DeepSeek dynamic router.
- flash-mHC's fused implementation is restricted to four streams, CUDA, and BF16; its fallback is separate.
- dmlx retains some mHC operations on CPU; the existence of Metal experiments does not make the entire HC path Metal-native.
- The RDNA patch set and Intel-XPU extension are independent implementations, not DeepSeek/Qwen official code. The XPU extension has recorded device correctness tests, but the broader offload system documents approximate expert masking and write-back race limitations.
- oHC and xHC project artifacts must not be called released code. xHC still had no implementation source on October 9; its authors report NeurIPS 2026 acceptance.
- No training jobs, remote code, or performance benchmarks were run in this audit. Source inspection and author-recorded test results are not independent reproduction.

An API health check of all 76 explicit README HC-path links found four moved paths. Current replacements were inspected and linked for **torchtitan-npu/torchtitan-npu**, **mindspore-ai/hyper-parallel**, **lightseekorg/tokenspeed**, and **tile-ai/tilelang-musa**. The MUSA repository's mHC examples still target CUDA/CUPTI; they are not described as verified MUSA execution. This path check is not a full re-review of every old codebase.

## Revised existing entries and official resources

- sHC is now **s²HC**, with current title *Spectral-Sphere-Constrained Hyper-Connections*, [arXiv v2](https://arxiv.org/abs/2603.20896) dated September 25 and author-reported NeurIPS acceptance. Its first-date row remains March 21; no duplicate publication is added.
- Original [HC / ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f1e8ff97057e0c3f6bcb7018c78d4df1-Abstract-Conference.html), [mHC / PMLR](https://proceedings.mlr.press/v306/xie26aa.html), and [KromHC / PMLR](https://proceedings.mlr.press/v306/zhou26k.html) are linked. The obsolete claim that mHC's proceedings were not indexed is removed.
- The model-resource section adds or expands official DeepSeek-V4.1-Flash, Tencent Hy4, GLM-5.3-Flash, Qwen3.8-Flash-Next, Cactus Needle 3/Whistle, and the author-linked Depth-Attention mHC-3B baseline. Quantizations are not counted as separate model families or papers.
- `pmetal-mhc` is labeled a crate index: docs.rs says 0.4.0 rustdoc failed. The page lists 0.5.0, but its current documentation build was not verified. PyPI `hyper-connections` remains 0.4.11 and conda-forge 0.2.1 at this audit.

## Unresolved primary-source candidates

These are discoverable candidates, not silently verified or counted publications.

| Candidate | What is known | Why it is pending |
|---|---|---|
| [uHC: Uncoupled Hyper-Connections](https://openreview.net/forum?id=Qnj7Lf8Bz2) | Cited by the multi-head report as an ICLR 2027 submission | OpenReview forum/PDF/API retrieval failed; browser stopped at verification. Primary date, authorship status, and full mechanism remain unverified. |
| [On the Design of Hyper-Connections: Simplification, Stability, and Scalability](https://openreview.net/forum?id=i2WyUVJUJ2) | Cited as an ICLR 2027 submission | Same primary-source access limitation; no confirmed chronological date or full-text assessment. |
| [One Marginal Is Enough: What Doubly Stochastic Routing Buys Hyper-Connections, and What It Costs](https://openreview.net/forum?id=1BC0eYN2uS) | Cited as an ICLR 2027 submission | Same limitation; do not infer an accepted venue or verified result from another author's references. |
| [Where a Hybrid MoE Spends Its Bytes: A 177B Engram Model with a 34.7 GiB GPU-Resident Core](https://www.badtheorylabs.com/papers/where-a-hybrid-moe-spends-its-bytes) — Al-ameen Olajide | Full author-hosted 8-page PDF read; §5 derives gated-HC scale migration and verifies it numerically, while excluding general rotations | Formal adjacent/analysis candidate. Only September 2026 is supplied; exact public day was not verified. Retained here instead of inventing a day in the strictly day-sorted table. No arXiv/DOI or independent review located. |
| *LLM-based renewable energy generation forecasting via multi-scale hyper-connection feature fusion*, DOI `10.1016/j.energy.2026.141683` | Publisher/abstract search candidate | Full mechanism not accessible in the prior sweep; residual-stream HC lineage remains unconfirmed. |

## Exclusions and deduplication in this sweep

- [Structured Residual Connectivity Matters for Diffusion Transformers](https://arxiv.org/abs/2609.33203) mentions HC/mHC/xHC in background/distinctions without an HC experiment; citation-only exclusion.
- `arjunmnath/MaRN` and `qingyichen/hammar` concern parameter-manifold training or motion planning, not HC persistent residual streams.
- `leapdragon/vllm-rdna2-qwen` is an upstream vLLM fork. `myllmbox/qwen38-flash-next-cluster-recipe` chiefly recipes/backports existing vLLM HC support; it is not counted as a separate HC implementation root.
- [fabiogreter/lily-qwen3.8-flash-next](https://github.com/fabiogreter/lily-qwen3.8-flash-next) and [beamivalice/sushi](https://github.com/beamivalice/sushi) contain substantive HC implementations, but their READMEs explicitly identify a fork or detached fork despite API `fork:false`. Under the current fork-deduplication policy they are retained here as discoverable divergent extensions, not added as independent canonical roots. This exclusion is an editorial scope choice, not a claim that the code is missing or trivial.
- Kimi K3 / Ember-1 Block Attention Residuals are different depth-routing methods, not mHC deployments.
- No verifiable author code for TEMPER was located; a search failure is not proof that no such code exists.

## Historical PaddleFleet row

The old `PaddlePaddle/PaddleFleet` root and HC path returned public API 404 during this audit. No redirect or official migration statement was verified; PaddleFleetX is not assumed to be its successor. The README keeps the old row explicitly marked historical, preserving its August 20 timestamp and 23 Stars rather than manufacturing current metadata.

The old captured commit's [HC source survives in the fork network](https://github.com/ForFishes/PaddleFleet-v0/blob/ceef5bdfbb792e9f2661df02581a73577f0e55ce/src/paddlefleet/transformer/hyper_connection.py). That personal repository is recovery evidence, not an official replacement or a newly counted upstream root.

## Search boundary

The October sweeps covered arXiv title/abstract/full-text variants, references and forward citations, OpenReview and proceedings, Zenodo and author-hosted manuscripts, GitHub repository/source search and current default-branch inspection, official Hugging Face cards/configs, model release notes, packages, and selected technical explainers. Queries included `Hyper-Connections`, `hyper connections`, `mHC`, `manifold-constrained`, `Birkhoff`, `residual streams`, method names, and DeepSeek/Qwen source classes. A separate October 4–9 delta pass searched new papers, repositories, and official resources.

Coverage remains bounded by public discoverability and access. Anonymous/renamed submissions, author websites without dates, unavailable publisher full text, private repositories, and unindexed source-only integrations can be missed. The defensible statement is **a dated, source-verified catalog with disclosed gaps**, not absolute completeness.
