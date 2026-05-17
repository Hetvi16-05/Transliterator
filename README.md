# Transliterator — Devanagari → Roman

A multi-model Hindi transliteration system trained on the Dakshina dataset.

## Project Structure

```
transliterator/
├── dataset/
│   ├── raw/                   # dakshina_hi.tsv (downloaded dataset)
│   ├── processed/             # train.csv, val.csv, test.csv
│   └── prepare_dataset.py     # Cleans & splits raw data
│
├── models/
│   ├── model1_rulebased.py    # Rule-based character mapping
│   ├── model2_rnn.py          # Vanilla RNN Seq2Seq
│   ├── model3_lstm.py         # LSTM Seq2Seq
│   ├── model4_bilstm.py       # Bidirectional LSTM Seq2Seq
│   ├── model5_gru.py          # GRU Seq2Seq
│   ├── model6_attention.py    # LSTM + Bahdanau Attention
│   ├── model7_transformer.py  # Transformer (from scratch)
│   └── model8_cnn_lstm.py     # CNN + LSTM hybrid
│
├── saved_models/              # .h5 weights for each trained model
├── evaluation/
│   ├── evaluate_all.py        # Runs all 8 models, prints WER/CER/Acc
│   └── results.csv            # Comparison table
│
├── static/                    # CSS & JS assets
├── templates/index.html       # Frontend UI
├── app.py                     # Flask entry point
├── predict.py                 # Load best model & run inference
└── train_all.py               # Train all models in one go
```

## Team

| Person | Responsibility |
|--------|---------------|
| Person 1 | `prepare_dataset.py`, `model1_rulebased.py`, `model2_rnn.py` |
| Person 2 | `model3_lstm.py`, `model4_bilstm.py` |
| Person 3 | `model5_gru.py`, `model6_attention.py`, `evaluate_all.py` |
| Person 4 | `model7_transformer.py`, `model8_cnn_lstm.py`, `app.py`, `index.html` |

## Setup

```bash
pip install -r requirements.txt
python train_all.py          # train all models
python evaluation/evaluate_all.py   # compare metrics
python app.py                # launch Flask UI
```

## Dataset

[Dakshina Dataset](https://github.com/google-research-datasets/dakshina) — Hindi lexicon (`hi/lexicons/hi.translit.sampled.train.tsv`)
