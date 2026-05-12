# Originality and Academic-Integrity Audit

## Scope

This audit compares the project files supplied for the final deliverable:

- `SSL_PatchTST.py` -> packaged as `code/ssl_patchtst.py`
- `PatchTST(4).py` -> cleaned and packaged as `code/patchtst_student_baseline.py`

against the official PatchTST paper authors' implementation at:

- https://github.com/yuqinie98/PatchTST
- key official files such as `PatchTST_supervised/layers/PatchTST_backbone.py` and `PatchTST_self_supervised/src/models/patchTST.py`

This is a practical academic-integrity review, not a formal legal opinion or plagiarism-system report.

## Expected overlap

Some overlap is unavoidable and acceptable because the assignment is a re-implementation of PatchTST. The following shared ideas are algorithmic content from the paper and should be cited:

- segmenting time series into patches;
- channel-independent encoding with shared weights across series;
- positional embeddings over patches;
- Transformer encoder layers;
- a flattened forecasting head;
- masked-patch reconstruction for SSL pretraining;
- evaluation on the Electricity long-term forecasting benchmark.

These should be described as coming from Nie et al., not as original model contributions.

## Findings for `code/ssl_patchtst.py`

Direct-copying risk appears **low**.

Evidence supporting originality:

1. **Different organization.** The project uses a single unified module containing tokenizers, encoder, SSL model, forecast head, evaluation, training, and checkpoint helpers. The official repo separates supervised and self-supervised code into different folders and uses different class boundaries.
2. **Different class/function names.** The official supervised code uses names such as `PatchTST_backbone`, `TSTiEncoder`, `Flatten_Head`, `_MultiheadAttention`, and `RevIN`. The project uses names such as `NoPadPatchTokenizer`, `EndPadPatchTokenizer`, `PatchTSTEncoder`, `PatchTSTSelfSupervised`, `FixedPatchTSTForecastHead`, `random_block_patch_mask`, and `train_forecaster_onecycle`.
3. **Different attention implementation.** The main project code uses PyTorch `nn.MultiheadAttention`; the official repo implements custom multi-head attention and scaled-dot-product attention modules, with optional residual attention.
4. **Added ablation logic.** The block masked-patch objective, realized-mask-ratio logging, fixed validation masks, unified fine-tuning loop, checkpoint JSON helpers, and quick experiment structure are project-specific additions.
5. **Tokenizer differences are explicit.** The project separates `NoPadPatchTokenizer` for SSL from `EndPadPatchTokenizer` for supervised PatchTST-style reproduction, which is not a direct copy of the official file layout.

Potential risk points:

- The architecture intentionally follows the paper, so the paper and official repository must be acknowledged.
- The code comments should avoid implying that patching/channel independence/SSL masking are novel inventions of this project.

## Findings for `code/patchtst_student_baseline.py`

Direct-copying risk appears **low to moderate**.

Evidence supporting originality:

1. It implements a minimal educational Transformer/PatchTST model with manually written `MultiHeadAttention`, `FeedForward`, `EncoderLayer`, and `PatchTST` classes.
2. It does not match the official repo's module hierarchy, `RevIN` implementation, residual-attention option, or official heads.
3. The original uploaded file contained course-style `TODO` comments and an unrelated BERT helper, which suggests it came from a learning scaffold rather than from the official PatchTST repository.

Potential risk points:

- Generic Transformer code has a very standard structure, so it can resemble many tutorials. This is usually acceptable if the source of any scaffold is acknowledged according to course policy.
- The cleaned file removes unrelated BERT code and TODO markers to reduce confusion for graders and future users.

## Recommended changes already made in this packaged deliverable

1. Renamed `SSL_PatchTST.py` to `code/ssl_patchtst.py` for import-friendly use.
2. Renamed and cleaned `PatchTST(4).py` to `code/patchtst_student_baseline.py`; removed unrelated BERT processing code from the packaged reference file.
3. Added `code/data.py` for reproducible Electricity CSV reading, train-only standardization, and sliding-window DataLoaders.
4. Added `code/run_experiment.py` with CLI presets for supervised, PatchTST/42, random SSL, block SSL, overlap SSL, and last-value runs.
5. Added `code/sanity_check.py` for a fast CPU forward/backward smoke test.
6. Added `requirements.txt`, `.gitignore`, `LICENSE`, `data/README.md`, `results/README.md`, and machine-readable `results/metrics_table.csv`.
7. Added explicit references to the PatchTST paper, official implementation, Electricity dataset, and PyTorch in `README.md` and the report.

## Recommended wording for submission

Use this wording or something close to it:

> This repository is an independent course-project re-implementation of PatchTST for the Electricity forecasting benchmark. The model design follows Nie et al. (ICLR 2023), and the official authors' repository was used as a reference for the experimental setting. The code here is organized independently for the reproduction and SSL ablation experiments; it does not import the official PatchTST repository.

Avoid wording such as:

> We introduce PatchTST.

or

> Our model proposes channel-independent patching.

Those claims belong to the original paper. Your original contribution is the controlled re-implementation, the masked-SSL ablation comparison, the block-mask variant, the overlap control, and the scale-sensitivity probe.

## License note

The official PatchTST repository is Apache-2.0. If any exact official code is copied into this repository, preserve the Apache-2.0 license notice and clearly mark the copied source. The packaged code here is treated as an independent implementation and is released under MIT for course-submission convenience.

## Bottom line

With the included acknowledgements and references, the risk of academic-integrity issues from direct copying appears low. The main requirement is to clearly frame the work as a re-implementation of Nie et al.'s PatchTST and to identify the project's own additions: masked SSL ablation, block masking, overlap control, code unification, and scale-sensitivity analysis.
