# PatchTST on Electricity: Reproduction, Masked SSL, and Scale Sensitivity

**Author:** Zhihan Liu, Cornell University  
**Course context:** Cornell CS 5782 Deep Learning final project  
**Repository:** https://github.com/KrasusZL/5782-Final-Project  
**Paper:** Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam, *A Time Series Is Worth 64 Words: Long-term Forecasting with Transformers*, ICLR 2023.

## 1. Introduction

This repository contains a course-project re-implementation of PatchTST on the Electricity forecasting benchmark. PatchTST represents each univariate channel as a sequence of fixed-length subseries patches, shares a Transformer backbone across channels, and uses either a supervised forecasting head or a masked-patch reconstruction head.

The project has two goals: reproduce the supervised PatchTST/64 Electricity-96 result, and test whether masked-patch self-supervised pretraining and temporal coarse-graining analysis reveal useful structure beyond final test MSE.

## 2. Chosen Result

The primary reproduction target is **PatchTST/64 on Electricity with prediction horizon 96**, reported in Table 3 of the PatchTST paper. The paper reports test MSE **0.129** and MAE **0.222** for this setting.

This result was chosen because it is a central supervised long-term forecasting benchmark for the paper's main claim: patching plus channel-independence can make Transformer forecasting competitive on large multivariate time series. The project also compares against the paper's SSL context from Table 4 and extends it with random and block masked-patch pretraining.

## 3. GitHub Contents

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── code/
│   ├── ssl_patchtst.py              # unified PatchTST-style SSL/forecasting implementation
│   ├── patchtst_student_baseline.py # cleaned student supervised baseline
│   ├── data.py                      # Electricity dataloader and chronological splits
│   ├── run_experiment.py            # command-line reproduction runner
│   └── sanity_check.py              # fast CPU smoke test
├── data/
│   ├── electricity.csv.gz           # compressed Electricity benchmark CSV
│   └── README.md
├── results/
│   ├── metrics_table.csv            # main result table used by poster/report
│   └── README.md
├── poster/
│   └── 5782_Poster.pdf
├── report/
│   ├── patchtst_electricity_report.pdf
│   ├── patchtst_electricity_report.tex
│   └── figures/
├── notebooks/
│   ├── PatchTST_exp_portable.ipynb  # optional Colab/local notebook wrapper
│   └── README.md
└── ORIGINALITY_AUDIT.md
```

## 4. Re-implementation Details

The main implementation is `code/ssl_patchtst.py`. It includes a channel-independent PatchTST-style encoder, no-pad and end-pad patch tokenizers, masked-patch reconstruction, supervised forecasting heads, MSE/MAE evaluation, OneCycle training loops, checkpointing, and JSON logging.

Main experiment configurations:

| Method | Lookback `L` | Horizon `H` | Patch `P` | Stride `S` | Pretrain | Fine-tune |
|---|---:|---:|---:|---:|---:|---:|
| Previous value | 512 | 96 | -- | -- | 0 | 0 |
| PatchTST/42 supervised | 336 | 96 | 16 | 8 | 0 | 30 |
| PatchTST/64 supervised | 512 | 96 | 16 | 8 | 0 | 30 |
| Random SSL+FT | 512 | 96 | 12 | 12 | 30 | 15 |
| Block SSL+FT | 512 | 96 | 12 | 12 | 30 | 10 |
| Overlap SSL control | 336 | 96 | 16 | 8 | 30 | 15 |

The original notebook workflow mounted a private Google Drive folder. That is **not** the canonical rerun path in this repository. The canonical path is the repo-relative command-line runner in `code/run_experiment.py`; the optional notebook in `notebooks/` simply wraps those same commands.

## 5. Reproduction Steps

### 5.1 Local setup

```bash
git clone https://github.com/KrasusZL/5782-Final-Project.git
cd 5782-Final-Project
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ is recommended. A CPU is enough for the smoke tests; a CUDA GPU is strongly recommended for full reproduction runs.

### 5.2 Sanity check

Run this first. It uses synthetic tensors and should finish on CPU in a few seconds.

```bash
python code/sanity_check.py
```

Expected output includes `"status": "ok"`, a finite SSL loss, a masked-patch prediction shape, and a forecast shape.

### 5.3 Check the dataset path

The repository tracks `data/electricity.csv.gz`, which pandas can read directly. Confirm that the dataloader and last-value baseline work:

```bash
python code/run_experiment.py \
  --experiment last_value \
  --data-path data/electricity.csv.gz \
  --device cpu
```

### 5.4 Full supervised PatchTST/64 reproduction

```bash
python code/run_experiment.py \
  --experiment supervised \
  --data-path data/electricity.csv.gz \
  --device cuda \
  --batch-size 16
```

The default supervised preset uses `L=512`, `H=96`, `P=16`, `S=8`, `d_model=128`, 16 attention heads, 3 encoder layers, and 30 epochs.

### 5.5 SSL ablations

Random masked-patch SSL plus fine-tuning:

```bash
python code/run_experiment.py \
  --experiment random_ssl \
  --data-path data/electricity.csv.gz \
  --device cuda \
  --batch-size 16
```

Contiguous block masked-patch SSL plus fine-tuning:

```bash
python code/run_experiment.py \
  --experiment block_ssl \
  --data-path data/electricity.csv.gz \
  --device cuda \
  --batch-size 16
```

Overlapping-patch SSL control:

```bash
python code/run_experiment.py \
  --experiment overlap_ssl \
  --data-path data/electricity.csv.gz \
  --device cuda \
  --batch-size 16
```

### 5.6 Quick CPU smoke run

Use `--quick` to test the training/evaluation pipeline with smaller dimensions and one epoch. This does **not** reproduce the final metrics.

```bash
python code/run_experiment.py \
  --experiment random_ssl \
  --data-path data/electricity.csv.gz \
  --quick \
  --device cpu
```

### 5.7 Colab and notebook rerun note

Do not make Google Drive mounting the default. The old pattern

```python
from google.colab import drive
drive.mount('/content/drive')
PROJECT_DIR = Path('/content/drive/MyDrive/...')
```

will fail for graders because they do not have your private Drive path. In Colab, use the public repo instead:

```python
!git clone https://github.com/KrasusZL/5782-Final-Project.git
%cd 5782-Final-Project
!pip install -r requirements.txt
```

Then make all paths relative to the repository root:

```python
from pathlib import Path
import sys

PROJECT_DIR = Path.cwd().resolve()
CODE_DIR = PROJECT_DIR / "code"
DATA_DIR = PROJECT_DIR / "data"
for p in [PROJECT_DIR, CODE_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATA_PATH = DATA_DIR / "electricity.csv.gz"
assert DATA_PATH.exists(), f"Missing dataset: {DATA_PATH}"
```

Robust imports for the packaged code are:

```python
from patchtst_student_baseline import PatchTST
import ssl_patchtst as ssl_ablation
from data import ElectricityDataConfig, make_electricity_dataloaders
```

If you keep a private Drive workflow for your own convenience, guard it with `USE_DRIVE = False` and never require it for the public rerun path.

### 5.8 Outputs

Each training run writes outputs under `results/runs/<timestamp>_<experiment>/`:

```text
config.json
metrics.json
forecast_model.pt
ssl_model.pt        # SSL experiments only
```

These run directories are intentionally ignored by Git because model checkpoints are large and machine-specific.

## 6. Results / Insights

![Forecasting gain over PatchTST/64 paper baseline](report/figures/forecasting_gain_user.png)

| Method | Test MSE | Gain vs. PatchTST/64 paper baseline |
|---|---:|---:|
| Previous-value baseline | 1.5880 | much worse |
| DLinear paper baseline | 0.1400 | -0.0110 MSE |
| PatchTST/42 supervised | 0.1326 | -0.0036 MSE |
| PatchTST/64 paper baseline | 0.1290 | reference |
| PatchTST/64 supervised reproduction | 0.1294 | near match |
| Overlap SSL control | 0.1284 | +0.0006 MSE |
| PatchTST/42 SSL+FT paper reference | 0.1260 | +0.0030 MSE |
| Random SSL+FT | 0.1239 | +0.0051 MSE |
| Block SSL+FT | **0.1237** | **+0.0053 MSE** |

The supervised PatchTST/64 reproduction nearly matches the paper result. Masked-patch SSL improves fine-tuned forecasting in these runs, and block masking is slightly better than independent random masking. The temporal coarse-graining probe suggests the learned representation is scale-structured rather than scale-invariant.

## 7. Conclusion

The repository reproduces the central Electricity-96 PatchTST/64 result within a small margin and extends the study with masked-patch SSL and scale-sensitivity tests. The most important implementation lesson is that patch count, padding convention, instance normalization, and data-split handling must be controlled carefully for the reproduction to be interpretable.

## 8. References

- Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. 2023. *A Time Series Is Worth 64 Words: Long-term Forecasting with Transformers*. ICLR 2023. https://arxiv.org/abs/2211.14730
- Official PatchTST implementation by the paper authors: https://github.com/yuqinie98/PatchTST
- Artur Trindade. 2015. *ElectricityLoadDiagrams20112014*. UCI Machine Learning Repository. https://doi.org/10.24432/C58C86

## 9. Acknowledgements

This work was completed as a Cornell CS 5782 Deep Learning final project. The project builds on the PatchTST paper and uses the Electricity benchmark from the UCI Machine Learning Repository.

## License and attribution note

The re-implementation code in this repository is released under the MIT License; see `LICENSE`. The PatchTST architecture and paper figures belong to the original PatchTST authors and are cited in the report. The Electricity dataset should be cited through the UCI Machine Learning Repository. The file `ORIGINALITY_AUDIT.md` summarizes the difference between this project code and the official PatchTST repository.
