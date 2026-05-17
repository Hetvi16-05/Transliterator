"""
model2_rnn.py
--------------
Model 2: Vanilla RNN Seq2Seq
- Simple Encoder-Decoder using basic SimpleRNN cells
- No attention, no bidirectionality
- This is the simplest neural model — expect lower accuracy

Person 1 is responsible for this file.

Usage:
    python models/model2_rnn.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from models.utils import build_vocab, encode_sequences, load_pairs, compute_metrics

# ──────────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────────
EMBEDDING_DIM = 64
UNITS         = 128
BATCH_SIZE    = 16
EPOCHS        = 80
MAX_ENC_LEN   = 30
MAX_DEC_LEN   = 30
MODEL_PATH    = 'saved_models/model2_rnn.keras'


# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────

def load_data():
    train_pairs = load_pairs('dataset/processed/train.csv')
    val_pairs   = load_pairs('dataset/processed/val.csv')
    test_pairs  = load_pairs('dataset/processed/test.csv')
    all_pairs   = train_pairs + val_pairs + test_pairs
    input2idx, idx2input, target2idx, idx2target = build_vocab(all_pairs)

    enc_tr, dec_in_tr, dec_out_tr = encode_sequences(train_pairs, input2idx, target2idx, MAX_ENC_LEN, MAX_DEC_LEN)
    enc_va, dec_in_va, dec_out_va = encode_sequences(val_pairs,   input2idx, target2idx, MAX_ENC_LEN, MAX_DEC_LEN)
    enc_te, dec_in_te, dec_out_te = encode_sequences(test_pairs,  input2idx, target2idx, MAX_ENC_LEN, MAX_DEC_LEN)

    return (enc_tr, dec_in_tr, dec_out_tr,
            enc_va, dec_in_va, dec_out_va,
            enc_te, dec_in_te, dec_out_te,
            input2idx, idx2input, target2idx, idx2target,
            train_pairs, test_pairs)


# ──────────────────────────────────────────────
# MODEL DEFINITION
# ──────────────────────────────────────────────

def build_model(input_vocab_size, target_vocab_size):
    # Encoder
    encoder_inputs = layers.Input(shape=(MAX_ENC_LEN,), name='encoder_input')
    enc_emb        = layers.Embedding(input_vocab_size, EMBEDDING_DIM, name='enc_embedding')(encoder_inputs)
    enc_out, enc_state = layers.SimpleRNN(UNITS, return_state=True, name='encoder_rnn')(enc_emb)

    # Decoder
    decoder_inputs = layers.Input(shape=(MAX_DEC_LEN,), name='decoder_input')
    dec_emb        = layers.Embedding(target_vocab_size, EMBEDDING_DIM, name='dec_embedding')(decoder_inputs)
    dec_out, _     = layers.SimpleRNN(UNITS, return_sequences=True, return_state=True,
                                      name='decoder_rnn')(dec_emb, initial_state=enc_state)
    outputs        = layers.Dense(target_vocab_size, activation='softmax', name='output')(dec_out)

    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ──────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────

def predict_word(model, roman, input2idx, idx2target, target2idx):
    enc_in = np.array([pad_seq([input2idx.get(c, 0) for c in roman.lower()], MAX_ENC_LEN)])
    dec_in = np.zeros((1, MAX_DEC_LEN), dtype=np.int32)
    dec_in[0, 0] = target2idx['<SOS>']
    result = ''
    for i in range(1, MAX_DEC_LEN):
        preds  = model.predict([enc_in, dec_in], verbose=0)
        tok_id = np.argmax(preds[0, i-1])
        char   = idx2target.get(tok_id, '')
        if char == '<EOS>' or char == '<PAD>':
            break
        result += char
        if i < MAX_DEC_LEN:
            dec_in[0, i] = tok_id
    return result


def pad_seq(seq, max_len):
    return seq[:max_len] + [0] * max(0, max_len - len(seq))


# ──────────────────────────────────────────────
# TRAIN + EVALUATE
# ──────────────────────────────────────────────

def train():
    print("\n🚀 Model 2: Vanilla RNN — Training...\n")
    (enc_tr, dec_in_tr, dec_out_tr,
     enc_va, dec_in_va, dec_out_va,
     enc_te, dec_in_te, dec_out_te,
     input2idx, idx2input, target2idx, idx2target,
     train_pairs, test_pairs) = load_data()

    model = build_model(len(input2idx), len(target2idx))
    model.summary()

    model.fit(
        [enc_tr, dec_in_tr], dec_out_tr,
        validation_data=([enc_va, dec_in_va], dec_out_va),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1
    )

    os.makedirs('saved_models', exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\n✅ Model saved → {MODEL_PATH}")

    # Evaluate
    preds = [predict_word(model, r, input2idx, idx2target, target2idx) for r, _ in test_pairs]
    refs  = [d for _, d in test_pairs]
    metrics = compute_metrics(preds, refs)
    print("\n📊 Model 2 — Vanilla RNN Results:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")
    return model, input2idx, idx2target, target2idx, metrics


if __name__ == '__main__':
    train()
