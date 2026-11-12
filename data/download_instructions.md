# Dataset Download & Setup Instructions

Follow one of the methods below to download the official dataset into the `data/` directory.

---

## Method 1: Kaggle Official CLI (Recommended)

1. **Install Kaggle CLI and Configure Credentials**:
   ```bash
   pip install kaggle
   ```
   Place your `kaggle.json` API token in `~/.kaggle/kaggle.json` (chmod 600).
   Ensure you have accepted the competition rules on the [Kaggle Competition Page](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/rules).

2. **Download Competition Files**:
   ```bash
   cd data/
   kaggle competitions download -c jigsaw-multilingual-toxic-comment-classification
   ```

3. **Extract the Files**:
   ```bash
   unzip jigsaw-multilingual-toxic-comment-classification.zip
   ```
   The key files extracted are:
   - `jigsaw-toxic-comment-train.csv`: Primary English training dataset (~223k rows)
   - `validation.csv`: Multilingual validation dataset (es, it, tr)
   - `test.csv`: Multilingual test dataset (es, it, tr, fr, ru, pt)

---

## Method 2: Hugging Face Datasets (Alternative)

If you prefer programmatic loading without Kaggle credentials, you can load the Jigsaw dataset or Wikipedia toxicity subsets via Hugging Face:

```python
from datasets import load_dataset

# English Jigsaw Toxic Comment Classification
dataset = load_dataset("jigsaw_toxicity_pred")
train_df = dataset["train"].to_pandas()
train_df.to_csv("data/jigsaw-toxic-comment-train.csv", index=False)
```

---

## Expected Directory Layout After Download

```
data/
├── README.md
├── download_instructions.md
├── synthetic_smoke_test.csv
├── jigsaw-toxic-comment-train.csv   # Raw English training data
├── validation.csv                   # Multilingual validation data
└── test.csv                         # Multilingual test data
```

---

## Running Quick Pipeline Verification (No Download Needed)

To verify the training, evaluation, prediction, and Streamlit pipeline without downloading the full dataset, you can immediately run:

```bash
python train.py --model baseline --use-synthetic
python evaluate.py --model baseline --use-synthetic
python predict.py --model baseline --text "This is a great, well-written article!"
streamlit run app.py
```
