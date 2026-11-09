# HC/mHC coverage audit — 2026-11-09

This incremental audit updates the [2026-10-09 snapshot](https://github.com/JiwenJ/Awesome-Hyper-Connection/tree/97d91c20e9d831a75f152402f849295ec3440d23), including the verified gaps from the October 15 sweep and the later October/November delta. This is a dated catalog, not a claim of web-wide exhaustiveness. See the [previous audit](search-audit-2026-10-09.md) for historical decisions.

| Index | October 9 | November 9 |
|---|---:|---:|
| Counted publications | 71 | 78 |
| Direct papers and formal manuscripts | 49 | 52 |
| Adjacent/adoption and formal comparisons | 22 | 26 |
| Indexed but not counted records | 3 | 3 |
| Training/research repository roots | 104 | 117 |
| Inference/serving repository roots | 42 | 58 |
| Total repository roots | 146 | 175 |

The 29 added roots include 12 training/research entries (five of them explicitly labeled project/model artifacts) and 17 inference/serving entries. FlashInfer's category move accounts for the difference between additions and net changes by block. The November 9 API refresh captures Stars and default-branch commits for 174 accessible roots plus one explicitly historical row.

## Verified publications

Dates are UTC for the first public version we could verify containing the relevant material. The table is global, not split by track.

| Date | Record | Evidence and classification |
|---|---|---|
| 2026-11-04 | [On the Design of Hyper-Connections: Simplification, Stability, and Scalability](https://arxiv.org/html/2611.05984v1) | Direct · SimpleHC. Identity transport, dynamic read/write gates and shared group-entry state. The largest tested width gives 2.4% extra step time versus the vanilla residual baseline. The roughly 31% reduction versus mHC is an estimate of saved activations at sublayer boundaries and inside HC operations, not peak device memory. The author project contains the report/assets, not released training code. |
| 2026-10-27 | [Think Twice: The Metis-1.6 Technical Report](https://arxiv.org/html/2610.37903v1) | Adoption · §3 uses four full-width residual streams, pass-conditioned controllers and eight Sinkhorn iterations. The separate LatentMoE FFN projects to 1,024 dimensions; it does not compress the persistent residual streams. |
| 2026-10-19 | [Manifold-Constrained Residual Adaptation with Frame-Adaptive Layer Aggregation for Speaker Verification](https://arxiv.org/html/2610.25892v1) | Direct · Separate follow-up to the August speaker mHC paper. MCRA partitions a fixed-width residual into feature groups, mixes with a Sinkhorn-normalized matrix and interpolates with a learned gate; the paper reports mixing-constraint and gate ablations. |
| 2026-10-13 | [Depth Memory](https://arxiv.org/html/2610.18809v1) | Adjacent · Unifies PreLN, mHC and AttnRes in a depth-memory framework; mHC uses four streams and 20 Sinkhorn iterations. Author code contains actual mHC training configs, operator wrappers and throughput benchmarks. |
| 2026-10-09 | [Multi-Stream Attention Residuals](https://arxiv.org/html/2610.13630) | Adjacent · 18B/28B MoE comparisons and an AttnRes+mHC ablation. Stores single-stream block summaries and expands them on demand, not persistent mHC-style state. Its 14:13 UTC submission followed the prior 03:31 UTC catalog commit. The project currently supplies paper and figures, not implementation source. |
| 2026-09-09 | [How Fragile Is Safety Alignment at Frontier Scale?](https://arxiv.org/html/2609.09793v1) | Adjacent · §§2.3, 4.3 and 5.3 analyze the four-stream GLM-5.3-Flash model and intervention behavior in the presence of mHC; not a new HC architecture. |
| 2026-07-21 | [Dual Attention Residuals](https://arxiv.org/html/2607.18730v1) | Direct · Uses reciprocal cross-stream depth retrieval, dynamic read/write gates, and a two-stream doubly stochastic transport for the partial state inside each block. The first version already reports mHC comparisons and stream diagnostics. The September revision is not a second paper. |

## Repository and resource changes

The [README](../README.md#repository-index--last-default-branch-commit) is the canonical index of repository names, inspected source links, whole-repository Stars and latest default-branch committer timestamps. Within each block the timestamp includes seconds for sorting, even though the displayed time is rounded to minutes. Ties use the canonical repository name. The refresh does not imply that every older project was re-run or its claims independently reproduced.

- The implementation additions cover author research (Depth Memory), independent training experiments (uHC, llama3-mhc, deepseek-lab and others), the Flash Depth Mixers autograd integration over TileKernels, vendor serving paths (DeepInfer, vLLM-ERNIE, RTP-LLM, MAX), model-specific serving plugins, CPU/TPU/SYCL/Rust runtimes, and forward-only accelerator kernels. The AscendC MhcPre entry is a partial pre operator, not a full Sinkhorn mHC implementation.
- uHC's public source uses groupwise dynamic reads and per-stream low-rank writes, but returns full-width residual streams; this is not evidence of compressed persistent residual state. Its code does not establish the OpenReview submission status.
- Existing xHC now links released Megatron-LM training code and Triton kernels. The [training-source commit](https://github.com/aHapBean/xHC/commit/322b9562c3c3397205516771e9ddce39acf28eb4) is dated October 13; author-provided curves are not independent reproduction. [NeurIPS's program](https://neurips.cc/Downloads/2026) lists xHC and s²HC.
- Existing FlashInfer moves from inference/serving to training/research without being counted twice. The [October 21 merge](https://github.com/flashinfer-ai/flashinfer/pull/6897) and [v0.7.3 release notes](https://flashinfer.ai/releases/) document Cake mHC forward, backward and recomputation for supported four-stream shapes.
- The API reported canonical transfers for `torchtitan-npu/torchtitan-npu` → `cann-framework/torchtitan-npu` and `Blaizzy/mlx-vlm` → `Neywa-Labs/mlx-vlm`. The surviving stable repository IDs are 1210038933 and 787462297, respectively. Neither transfer adds a new root.
- The old Needle repository is now a redirect README; the current Cactus project carries the mHC and finetuning sources and replaces it in the index. Three other moved HC paths were updated: TileOPs' `_ops` module, gLLM's `python/layers/ops/deepseek_v4` Triton path, and the RDNA patch set's block 11 Qwen hyper-connection patch. All 101 explicit README `HC path` links resolved against the captured references after these repairs.
- The historical PaddleFleet root still returned public API 404. Its archived metadata remains explicitly historical; no official migration to PaddleFleetX is inferred. The [old source in the fork network](https://github.com/ForFishes/PaddleFleet-v0/blob/ceef5bdfbb792e9f2661df02581a73577f0e55ce/src/paddlefleet/transformer/hyper_connection.py) is recovery evidence, not a second upstream root.
- Model cards, checkpoint variants, package pages, technical notes and deployment guides remain separate from the paper count. Key additions include ERNIE-5.5-Flash-Base, SenseNova-Loop-9B-Preview, Raon-1 mini preview, Xing4.0-29B-A4B, the DeepSeek-V4.1-Flash-1017 checkpoint, and guides or release notes from GPUStack, vLLM, AMD, NVIDIA and Modular. For ERNIE the 2,048-dimensional latent expert path and the four-stream mHC residual path are distinct.

## Pending — not counted as publications

| Candidate | Verified so far | Still needed |
|---|---|---|
| [uHC: Uncoupled Hyper-Connections](https://openreview.net/forum?id=Qnj7Lf8Bz2) | Public training repository with distinct code. | OpenReview primary text and confirmed first-public date/submission status; the forum was gated by browser verification. |
| [One Marginal Is Enough: What Doubly Stochastic Routing Buys Hyper-Connections, and What It Costs](https://openreview.net/forum?id=1BC0eYN2uS) | Title and forum ID in author references. | Accessible manuscript, date, status and mechanism. |
| [Understanding the Effects of Hyper-Connections on Self-Attention Dynamics: A Bifurcation Analysis](https://neurips.cc/virtual/2026/poster/151454) | Title in the [official NeurIPS program](https://neurips.cc/Downloads/2026); [author publication list](https://tanmnguyen89.github.io/) names Nhi Y. Pham, Dung V. Nguyen, Trinh T. Nguyen, Thieu N. Vo and Tan M. Nguyen. | Full text and the first public day for the chronological table. |
| [Where a Hybrid MoE Spends Its Bytes](https://www.badtheorylabs.com/papers/where-a-hybrid-moe-spends-its-bytes) | Author manuscript with a relevant derivation in §5. | First public day; only September 2026 is verified. |
| [LLM-based renewable energy generation forecasting via multi-scale hyper-connection feature fusion](https://doi.org/10.1016/j.energy.2026.141683) | Publisher/abstract candidate. | Full text confirming residual-stream HC lineage. |
| [MoME: Confidence-Based Adaptive Routing over Quantized Latent Experts](https://doi.org/10.13140/RG.2.2.29270.61766) | Author-uploaded manuscript candidate with HC in its design. | Verify whether the full manuscript reports completed experiments or proofs, rather than design/accounting only. |
| [Modular Connections](https://selen-suyue.github.io/modular-connections/en/index.html) | Author note and microbenchmarks; the reported full-scale run was about 14% complete. | Formal archive/source verification and completed matched training comparison. Included as a technical note, not evidence of final iso-quality gains. |

The title previously linked to [OpenReview i2WyUVJUJ2](https://openreview.net/forum?id=i2WyUVJUJ2) is now included once via SimpleHC's independently readable arXiv v1. This does not assert a verified OpenReview venue decision or backdate the entry.

## Exclusions and evidence boundary

- [M3Former](https://arxiv.org/html/2611.02436v1) compares HC/mHC in its architectural taxonomy but proposes depth-wise role/group retrieval and does not report a substantive HC/mHC baseline. It is not added to the paper count.
- Surogate has a verified mHC serving path; a dedicated mHC training/backward path was not established in this sweep. SGLang AMD Mono's V4 mHC provenance points to already indexed ROCm code; no new independent implementation is asserted. The Depth Memory project links two mHC checkpoints, but their model cards were not independently readable in this sweep.
- Project pages carrying only paper, plots, or deployment documentation are labeled as artifacts. Downstream integrations that reuse upstream kernels are not represented as independent kernel implementations. Pure mirrors, detached forks with no material independent contribution, and synthetic benchmark placeholders do not add canonical roots.
- The October 15 and November sweeps searched arXiv title/abstract/full text, citation and venue sources, OpenReview, author archives, GitHub repositories and code, model cards/configs, package indexes, official release notes and technical guides. Candidate source paths were read for substantive implementations. Coverage is limited by discoverability, access gates and the distinction between a mention and meaningful HC usage.
- No training jobs, model downloads, or performance benchmarks were run for this update. Reported quality, runtime and memory results retain their authors' protocols and scope.

## Audit method and known exclusions

The publication and resource-content audit through 2026-11-09 used entity-level deduplication and searched:

1. arXiv title/abstract/full-text variants for `Hyper-Connections`, `hyper connections`, `mHC`, `manifold-constrained`, `Birkhoff`, and `residual streams`;
2. forward citation networks of the original HC and mHC papers, followed by primary-source verification;
3. OpenReview, PMLR, IEEE/Crossref, Zenodo, institutional publication pages, and author/project pages;
4. GitHub repository and source-code search for paper titles, method names, common class names, kernels, and DeepSeek-V4 integrations;
5. Hugging Face Hub collections/models/Spaces plus PyPI, conda-forge, crates/docs.rs, technical documentation, blogs, videos, and discussion indexes for non-paper resources;
6. manual deduplication of arXiv/venue versions, model quantizations, forks, renamed repositories, vendored framework copies, and successor repositories;
7. incremental post-2026-08-12, post-2026-08-14, and 2026-08-19–25 sweeps of new arXiv records, Crossref/publisher records, full-text HC/mHC comparison hits, GitHub repositories, source paths, model artifacts, and open integration PRs, with candidate repository files inspected on their current default branches;
8. a 2026-08-20 06:35 UTC refresh of the then-indexed 119 GitHub roots, followed by a 2026-08-25 API health/path validation and metadata capture for the 12 added roots; those older snapshots were superseded by the accessible-root refresh below;
9. a dedicated dataset/resource pass; no HC/mHC-specific standalone dataset was located, so general corpora such as FineWeb-Edu, TinyStories, and CIFAR are not mislabeled as HC resources;
10. 2026-10-04 and 2026-10-09 incremental sweeps of arXiv/full text, author references, formal manuscripts, proceedings, GitHub default-branch sources, official model cards/configurations, release notes, packages, and technical explainers; nine verified formal records and fifteen repository roots were added. Renamed paper versions and proceedings are updated without double-counting;
11. the October 15 and November 9 sweeps, with seven additional source-verified publications and 29 new canonical repository roots, including separately labeled documentation/project artifacts. The November API refresh covers every accessible indexed root; provenance, transfers, and exclusions are recorded above.

For revised papers, **First public date** means the first publicly verifiable version that actually contains the HC/mHC-relevant material. This is why mHC-Diff is dated from arXiv v2 and the recurrence-scaling case study from v2, while FlowNet uses its earlier NeurIPS/OpenReview publication date rather than its later arXiv deposit.

Publication rows aim for source-verified coverage under the policy above. Repositories and other artifacts are intentionally selective: inclusion favors upstream or author-linked projects, runnable implementations, substantive tests/benchmarks, stable model artifacts, and useful technical documentation. A GitHub keyword hit alone is insufficient.

Not counted under the paper policy:

- [Attention Residuals](https://arxiv.org/abs/2603.15031), [SiameseNorm](https://arxiv.org/abs/2602.08064), [Multi-Gate Residuals](https://arxiv.org/abs/2605.23259), and [Diffusion-Adaptive Routing](https://arxiv.org/abs/2605.20708) are related residual-routing work, but do not develop or materially apply the HC/mHC residual-stream mechanism.
- [Low-Rank Attention Residuals](https://arxiv.org/abs/2607.09694) compresses depth-routing keys while retaining full-width residual values and does not report an HC/mHC comparison; it is therefore an Attention-Residuals result, not low-rank HC state compression.
- [Intern-S2-Mobius](https://arxiv.org/abs/2608.14290), [Tight Sinkhorn Convergence Rates](https://arxiv.org/abs/2608.11760), and [Riemannian Attention](https://arxiv.org/abs/2608.01283) mention HC/mHC only as prior work or an application example; none implements or formally evaluates the HC residual-stream mechanism.
- [WhiteMatter](https://arxiv.org/abs/2608.18486) mentions HC/mHC only in related work and reports no HC/mHC implementation, baseline, or ablation. [V-REX](https://arxiv.org/abs/2608.20069) explicitly uses a standard decoder-only Transformer while forgoing architectural upgrades including mHC. Both are citation-only false positives from the 2026-08-19–25 sweep.
- [Structured Residual Connectivity Matters for Diffusion Transformers](https://arxiv.org/abs/2609.33203) discusses HC/mHC/xHC in background and method distinctions without an HC experiment; it is a citation-only result, unlike SiHC's direct intervention study.
- [Deep Delta Learning](https://arxiv.org/abs/2601.00417) is an input to later hybrid proposals, not itself an HC paper.
- [TextResNet](https://arxiv.org/abs/2602.08306) maps mHC-inspired stability ideas into discrete textual optimization rather than implementing the neural residual-stream mechanism; it is kept outside the publication count under the present scope.
- *Testing the Platonic Representation Hypothesis via Representation-Controlled Tokenization* uses “Harmonized Hyper-Connections” for bounded coupling among neighboring token representations, not Zhu/Xie-style persistent residual streams; it is an unrelated same-name method.
- The workshop construction later published as [Transformers Provably Learn to Internalize Chain-of-Thought](https://arxiv.org/abs/2605.28600) uses a single-stream position-wise Highway/GTrXL-style gate and explicitly distinguishes it from HC in the later text; the earlier “position-wise Hyper-Connections” label is therefore not treated as HC-family evidence.
- *An Application of Manifold-Constrained Hyper-Connection in a PWA for Sovereign Debt Restructuring* uses “mHC” for a learned latent representation manifold rather than the Birkhoff-constrained parallel residual-stream mechanism.
- *Manifold-Constrained Hyper-Connections: A Geometric Framework for Revolutionizing AI Training Stability* (DOI `10.13140/RG.2.2.26203.22568`) uses a different hypernetwork/manifold-weight-generation concept and is a terminology false positive.
- `gm24med/MHC` and its `mhc` package use a sliding history-window skip over prior layer states rather than Zhu/DeepSeek-style persistent parallel residual streams, so they are not listed as paper-faithful reference implementations.
- `thainamhoang/muonHC` describes a planned Hyperloop-mHC climate experiment, but its README explicitly says the mHC module still requires implementation; it is omitted as a placeholder at this audit date.
- `stephane-hairy/rlm-mhc` expands and contracts temporary flows inside a layer and then returns to a single residual stream, rather than maintaining the paper's persistent parallel residual state; it is therefore not listed as an mHC reference implementation.
- `WYH302/Manifold-Constrained-Hyper-Connections-and-Prompting-for-SAM-in-Camouflaged-Scene-Segmentation` applies Birkhoff mixing inside a temporary adapter and immediately collapses to one stream without the full persistent HC update, so it is omitted under the same mechanism boundary.
- `wtoth/hyper-connections` is an unfinished HC attempt: core forward paths contain placeholders and incompatible constructor calls, so it is not listed as a runnable reference implementation.
- The public MUDDFormer code contains MUDD but no HC baseline; the public IntTravel repository contains only data/preprocessing; and the Engram demo explicitly mocks its mHC backbone. Their papers remain relevant, but those repositories are not represented as HC implementations.
- `flagos-ai/FlagGems-vllm`, `dphnAI/sonar`, and `rednote-machine-learning/RedKnot` only synchronize or vendor HC code from already indexed FlagGems, vLLM, or SGLang roots. `aeonmindai/arc` labels its unconnected DeepSeek-V4 mHC file as work in progress; `Mesh-LLM/mesh-llm` is a lockfile-only match; and `ZhangDanyang-AMD/Lumen-RL` only invokes the already indexed Megatron Bridge. They are excluded rather than counted as independent HC implementations.
- Forks, development predecessors, review snapshots, and exact-content copies found around the tokenbender, lucidrains, KromHC, mHC-GNN, mHC-PEFT, and vLLM-Ascend resources are deduplicated to the most relevant upstream or publication repository. The paper-linked `whale-agent-lab/colibri` fork is the stated exception because the exact runtime snapshot is part of the numerical experiment; its generic upstream is not listed as a second HC repository.
- [Orthostochastic Residual Mixing for Manifold-Constrained Hyper-Connections](https://analemma.ai/papers/523b5fe7-e78c-4d60-ac87-e4b164d85f4a/) and [Range-Capped Sinkhorn for Reliable Manifold-Constrained Hyper-Connections](https://analemma.ai/papers/28c3de0a-db8b-464d-a8e5-f4099b1ceba7/) are explicitly generated by an automated research system. They are linked for traceability but excluded from the publication count pending stable authorship and independent verification.
