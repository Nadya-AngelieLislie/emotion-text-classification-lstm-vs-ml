# Emotion Text Classification: LSTM vs Classical ML

> Classifying text as anger, joy or fear, comparing an LSTM on Word2Vec embeddings with TF-IDF models.

## Overview

- Individual project comparing a deep learning model with classical machine learning for 3-class emotion classification of short English texts.
- Tests whether a more complex LSTM beats simple TF-IDF models on a small dataset.

## Data / Scope

- 5,937 labeled comments: anger (2,000), joy (2,000) and fear (1,937); 80/20 stratified split (1,188 test texts).
- The dataset file (`Emotion_classify_Data.csv`) is not included.

## Workflow

1. Cleaning: lowercase, remove digits and punctuation, collapse whitespace.
2. Tokenization with NLTK and stopword removal.
3. Deep learning: skip-gram Word2Vec (100 dimensions) used to initialize an Embedding layer, followed by LSTM(64), batch normalization and a 3-class output, trained 10 epochs with Adam.
4. Classical ML: TF-IDF (5,000 features) with Logistic Regression and Random Forest (100 trees).
5. Compared all models on accuracy, precision, recall and F1.

## Results

- Logistic Regression + TF-IDF was best: 92.2% accuracy (macro F1 0.92).
- Random Forest + TF-IDF: 91.5% accuracy; LSTM: 87% accuracy.
- The LSTM was strong on anger (F1 0.93) but weaker on joy (recall 0.75) and fear (precision 0.75).
- Takeaway: on about 6,000 short texts, classical models outperformed the deep learning model.

## Tech Stack

Python, TensorFlow/Keras, Gensim (Word2Vec), NLTK, scikit-learn

## Repository Structure

```
notebook/    Full notebook with code and outputs
figures/     Accuracy and per-class F1 charts (drawn from the notebook's reported results)
requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook/Emotion_Text_Classification_LSTM_vs_ML.ipynb
```

The notebook was written in Google Colab and reads the CSV from Google Drive; change the path in the data-loading cell.

## Team

Individual project by Nadya Angelie Lislie (BINUS University).

## Notes

- Student ID numbers were removed from the notebook text.
- TODO: add the course name this was submitted for.

## Author

Nadya Angelie Lislie, BINUS University (Computer Science and Statistics). [LinkedIn](https://www.linkedin.com/in/nadya-angelie-lislie/) | [GitHub](https://github.com/Nadya-AngelieLislie)

## Experiment Tracking

`src/experiment_tracking.py` provides an optional MLflow wrapper (`track_run`) for logging parameters, metrics and artifacts; it does nothing when MLflow is not installed. Run `mlflow ui` after logging runs to compare experiments.

## Continuous Integration

GitHub Actions (`.github/workflows/ci.yml`) runs `ruff` and `pytest` repository-health checks on every push; run them locally with `make lint test`.
