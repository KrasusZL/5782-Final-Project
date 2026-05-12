# Dataset Instructions

The `Electricity` dataset used in this project is not included in this repository because the raw CSV file is too large for normal GitHub storage.

## Dataset used

This project uses the standard `electricity.csv` dataset used in long-term time-series forecasting benchmarks, including PatchTST/Autoformer-style experiments.

Expected local file path:

```bash
data/electricity.csv
```

If your local training script expects another path, for example `./dataset/electricity.csv`, either change the script/configuration to point to `./data/electricity.csv`, or place the downloaded file in the expected directory.

## How to obtain the dataset

1. Download the benchmark datasets from the Autoformer/PatchTST dataset Google Drive:

   https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy

2. Locate the Electricity dataset file, usually named:

```bash
electricity.csv
```

3. Place the file in this repository as:

```bash
data/electricity.csv
```

After downloading, the repository should look like:

```text
5782-Final-Project/
├── README.md
├── code/
├── data/
│   ├── README.md
│   └── electricity.csv        # downloaded manually, not committed to GitHub
├── results/
├── poster/
└── report/
```

## Reproducibility note

The submitted code assumes that the Electricity dataset has been downloaded manually and placed at:

```bash
data/electricity.csv
```

The dataset file itself is excluded from the GitHub repository because of its large size.

## Suggested `.gitignore` entry

To avoid accidentally committing large dataset files, add the following lines to the top-level `.gitignore` file:

```gitignore
data/electricity.csv
data/*.csv
```
