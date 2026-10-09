# IMDB Movie Review Sentiment Classification (Bidirectional LSTM)

A binary text classifier that predicts whether a movie review is **positive** or **negative**, trained on 50,000 IMDB reviews with a Bidirectional LSTM in TensorFlow/Keras.

## Results

| Metric | Value |
|---|---|
| Best validation accuracy | 87.90% (epoch 3) |
| Training accuracy at that epoch | 90.20% |
| Validation loss at that epoch | 0.3241 |
| Training stopped | Epoch 8 of 20 (early stopping, best weights restored) |

After epoch 3 training accuracy kept rising (up to 95.47%) while validation accuracy fell slightly, a sign of overfitting; early stopping restored the epoch 3 weights.

Example predictions: "this movie is really cool!" is predicted positive, and "i dont like this movie because it's boring" is predicted negative.

## Dataset

IMDB Dataset of 50K Movie Reviews (`IMDB Dataset.csv`, included): 25,000 positive and 25,000 negative reviews. After cleaning, a review has 118.6 words on average (median 88) and the corpus has 236,145 unique words.

## Approach

1. **Labels:** `positive` = 1, `negative` = 0.
2. **Cleaning (NLTK):** remove HTML tags and non-letters, tokenize, drop English stopwords, lemmatize verbs with WordNet.
3. **Split:** 80/20 (40,000 / 10,000), `random_state=42`.
4. **Tokenization:** Keras `Tokenizer` with a 25,000-word vocabulary and `<oov>` token; sequences padded or cut to 100 tokens (about the median length).
5. **Model:**
   ```
   Embedding(25000, 16, input_length=100)
   Bidirectional(LSTM(16, dropout=0.5, return_sequences=True))
   GlobalMaxPool1D → Dropout(0.5)
   Dense(64, relu, L2 0.01) → Dropout(0.5)
   Dense(1, sigmoid)
   ```
6. **Training:** Adam (lr 0.001), binary cross-entropy, batch 100, up to 20 epochs; early stopping on validation accuracy (patience 5) plus a callback that stops at 90% train and validation accuracy.
7. **Evaluation:** accuracy and loss curves, predictions on new reviews.

## Tech Stack

Python, TensorFlow/Keras, NLTK, scikit-learn, pandas, Matplotlib, Google Colab.

## Project Structure

```
NLP-BinaryClassification/
├── IMDB Dataset.csv
└── ProyekNLP-Harry/
    ├── ProyekNLP-Harry.ipynb   # Notebook with outputs
    └── ProyekNLP-Harry.py      # Script exported from Colab
```

## Getting Started

```bash
git clone https://github.com/harrymardika/NLP-BinaryClassification.git
cd NLP-BinaryClassification
pip install tensorflow nltk scikit-learn pandas matplotlib jupyter
cp "IMDB Dataset.csv" ProyekNLP-Harry/    # the notebook reads the CSV from its own folder
jupyter notebook ProyekNLP-Harry/ProyekNLP-Harry.ipynb
```

The first run downloads the NLTK `punkt`, `wordnet`, and `stopwords` data.

## Limitations

- The tokenizer is fitted on both training and test reviews, so the test set influences the vocabulary.
- The validation set is also the test set and is used for early stopping.

## Author

**Harry Mardika** · [GitHub](https://github.com/harrymardika)
