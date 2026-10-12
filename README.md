# Fake News Detection with TF–IDF and Logistic Regression

This project explores a classical text-classification pipeline for distinguishing fake-news examples from other news text. It uses TF–IDF features and logistic regression as a supervised baseline.

## Method

The notebook follows a typical text-classification workflow:

1. Load and inspect the dataset.
2. Prepare the text and target labels.
3. Transform text into TF–IDF features.
4. Fit a logistic-regression classifier.
5. Evaluate predictions using classification metrics.

## Repository contents

| Path | Description |
| --- | --- |
| Fake_News_Detection.ipynb | Main notebook containing preprocessing, model training, and evaluation |

## How to run

1. Open the notebook in Jupyter Notebook, JupyterLab, or Google Colab.
2. Review the dataset-loading cell and provide the expected input data.
3. Install the libraries imported by the notebook.
4. Run the notebook from top to bottom.

The repository currently contains the notebook but no separate dataset file or pinned environment specification. Exact reproduction depends on identifying the original data source and package versions.

## Evaluation and limitations

The original notebook reports accuracy above 99% on its dataset. This should be interpreted as a result for that particular experimental setup, not as evidence that the model generalizes to new sources of news. Before using the score as a headline result, verify the split strategy, class balance, duplicate/near-duplicate records, and possible data leakage. A stronger evaluation would include precision, recall, F1, a confusion matrix, and a clearly held-out test set.

TF–IDF plus logistic regression is a useful baseline, but it may learn dataset- or publisher-specific patterns instead of factual reliability. Classification output should not be treated as a definitive fact-check.

## Future improvements

- Document the dataset, license, target labels, and preprocessing decisions.
- Use a reproducible train/validation/test split and fixed random seeds.
- Add a pinned dependency file.
- Report per-class precision, recall, and F1 alongside accuracy.
- Add error analysis with representative false positives and false negatives.
- Compare with a simple majority-class baseline and document limitations.

## Technologies

Python · Jupyter Notebook · TF–IDF · Logistic Regression · scikit-learn · Text Classification
