# Code

`PatchTST_exp.ipynb` is the main experiment notebook. It is kept in the original notebook workflow rather than rewritten into a standalone script.

Required notebook imports:

- `PatchTST.py`: supervised PatchTST implementation.
- `SSL_PatchTST.py`: masked self-supervised learning and fine-tuning utilities.
- `electricity_dataset.py`: dataset helper providing `prepare_splits` and `load_splits`.

For a local rerun, start Jupyter from this directory:

```bash
cd code
jupyter notebook PatchTST_exp.ipynb
```

Before running, make sure the dataset exists at `code/dataset/electricity.csv` from the repository root, or `dataset/electricity.csv` relative to this directory.
