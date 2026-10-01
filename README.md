# Sentiment Analysis using BERT

Fine-tuning `bert-base-uncased` on the IMDB movie review dataset to classify reviews as **positive** or **negative**.

## Overview
- **Task:** Binary sentiment classification
- **Model:** BERT (Bidirectional Encoder Representations from Transformers), `bert-base-uncased`
- **Dataset:** [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) (25k train / 25k test reviews). A random subset is used for speed: 4,000 train, 1,000 test.
- **Framework:** Hugging Face Transformers + PyTorch, run on Google Colab (T4 GPU)

## Approach
1. **Tokenization:** WordPiece tokenizer, truncation and padding to 256 tokens
2. **Model:** pretrained BERT plus a linear classification head (2 outputs) on the `[CLS]` representation
3. **Fine-tuning:** 2 epochs, batch size 16, learning rate 2e-5, AdamW, mixed precision (fp16)
4. **Evaluation:** accuracy, classification report, confusion matrix

## Results
| Metric | Value |
|---|---|
| Test accuracy | <your accuracy> |
| Precision / Recall / F1 | <from classification report> |

Confusion matrix and sample predictions are in the notebook.

## How to Run
1. Open `BERT_Sentiment_Analysis.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set **Runtime → Change runtime type → T4 GPU**
3. Run all cells in order

Dependencies (installed in the first cell): `transformers`, `datasets`, `evaluate`, `accelerate`, `scikit-learn`, `matplotlib`

## Example Predictions
| Review | Prediction |
|---|---|
| "I loved this movie, absolutely brilliant!" | Positive |
| "Total waste of time, awful acting." | Negative |

## Files
- `BERT_Sentiment_Analysis.ipynb`: full code, outputs, and explanations
