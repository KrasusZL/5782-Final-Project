# Dataset Instructions

This project uses the Electricity benchmark CSV used in long-term forecasting experiments.

The notebook expects the local path:

```text
code/dataset/electricity.csv
```

If `data/electricity.csv.gz` is included, expand it from the repository root:

```bash
mkdir -p code/dataset
gunzip -c data/electricity.csv.gz > code/dataset/electricity.csv
```

If the compressed file is not included, download `electricity.csv` from the standard Autoformer/PatchTST benchmark dataset release and place it at `code/dataset/electricity.csv`.

The uncompressed CSV is ignored by Git to avoid committing large local data copies.
