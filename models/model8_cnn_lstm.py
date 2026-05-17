"""
model8_cnn_lstm.py
-------------------
Model 8: CNN + LSTM Hybrid Seq2Seq
- CNN (1D Convolutions) extracts local n-gram features from character sequences
- LSTM then models long-range dependencies on top of CNN features
- Best of both worlds: local pattern detection + sequential memory

Architecture for viva:
  Input chars → Embedding → Conv1D (n-gram features) → MaxPool
              → LSTM encoder → LSTM decoder → Output chars

Why CNN before LSTM?
  CNN captures "sh", "kh", "bh" type digraph patterns efficiently
  LSTM then handles word-level sequential patterns

Person 4 is responsible for this file.

Usage:
    python models/model8_cnn_lstm.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from models.utils import (build_vocab, encode_sequences, load_pairs,
                           compute_metrics, configure_gpu)

EMBEDDING_DIM  = 64
CNN_FILTERS    = 128
KERNEL_SIZE    = 3     # trigram features
LSTM_UNITS     = 256
BATCH_SIZE     = 64
EPOCHS         = 100
MAX_ENC_LEN    = 30
MAX_DEC_LEN    = 30
MODEL_PATH     = 'saved_models/model8_cnn_lstm.keras'


def load_data():
    train_pairs = load_pairs(split='train')
    val_pairs   = load_pairs(split='val')
    test_pairs  = load_pairs(split='test')
    all_pairs   = train_pairs + val_pairs + test_pairs
    input2idx, idx2input, target2idx, idx2target = build_vocab(all_pairs)
    enc_tr, dec_in_tr, dec_out_tr = encode_sequences(train_pairs, input2idx, target2idx, MAX_ENC_LEN, MAX_DEC_LEN)
    enc_va, dec_in_va, dec_out_va = encode_sequences(val_pairs,   input2idx, target2idx, MAX_ENC_LEN, MAX_DEC_LEN)
    return (enc_tr, dec_in_tr, dec_out_tr,
            enc_va, dec_in_va, dec_out_va,
            input2idx, idx2input, target2idx, idx2target,
            train_pairs, test_pairs)


def build_model(input_vocab_size, target_vocab_size):
    # ── Encoder: Embedding → CNN → LSTM ──
    enc_input = layers.Input(shape=(MAX_ENC_LEN,), name='enc_input')
    enc_emb   = layers.Embedding(input_vocab_size, EMBEDDING_DIM)(enc_input)

    # CNN layers — extract local character n-gram patterns
    cnn1      = layers.Conv1D(CNN_FILTERS, KERNEL_SIZE, padding='same',
                               activation='relu', name='cnn1')(enc_emb)
    cnn2      = layers.Conv1D(CNN_FILTERS, KERNEL_SIZE, padding='same',
                               activation='relu', name='cnn2')(cnn1)
    # Residual connection: skip cnn1 → add to cnn2
    cnn_out   = layers.Add()([cnn1, cnn2])
    cnn_out   = layers.LayerNormalization()(cnn_out)

    # LSTM on top of CNN features
    enc_out, enc_h, enc_c = layers.LSTM(LSTM_UNITS, return_sequences=True,
                                          return_state=True, name='enc_lstm')(cnn_out)

    # ── Decoder: Embedding → LSTM ──
    dec_input = layers.Input(shape=(MAX_DEC_LEN,), name='dec_input')
    dec_emb   = layers.Embedding(target_vocab_size, EMBEDDING_DIM)(dec_input)
    dec_out, _, _ = layers.LSTM(LSTM_UNITS, return_sequences=True,
                                  return_state=True, name='dec_lstm')(
                                  dec_emb, initial_state=[enc_h, enc_c])

    # Optional: simple dot-product attention over encoder outputs
    attn      = layers.Attention(name='dot_attention')([dec_out, enc_out])
    dec_final = layers.Concatenate()([dec_out, attn])
    dec_final = layers.Dense(LSTM_UNITS, activation='tanh')(dec_final)

    outputs   = layers.Dense(target_vocab_size, activation='softmax')(dec_final)

    model = Model([enc_input, dec_input], outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def pad_seq(seq, max_len):
    return seq[:max_len] + [0] * max(0, max_len - len(seq))


def predict_word(model, roman, input2idx, idx2target, target2idx):
    enc_in = np.array([pad_seq([input2idx.get(c, 0) for c in roman.lower()], MAX_ENC_LEN)])
    dec_in = np.zeros((1, MAX_DEC_LEN), dtype=np.int32)
    dec_in[0, 0] = target2idx['<SOS>']
    result = ''
    for i in range(1, MAX_DEC_LEN):
        preds  = model.predict([enc_in, dec_in], verbose=0)
        tok_id = np.argmax(preds[0, i-1])
        char   = idx2target.get(tok_id, '')
        if char in ('<EOS>', '<PAD>'):
            break
        result += char
        if i < MAX_DEC_LEN:
            dec_in[0, i] = tok_id
    return result


def train():
    configure_gpu()
    print("\n🚀 Model 8: CNN + LSTM Hybrid — Training...\n")
    (enc_tr, dec_in_tr, dec_out_tr,
     enc_va, dec_in_va, dec_out_va,
     input2idx, idx2input, target2idx, idx2target,
     train_pairs, test_pairs) = load_data()

    model = build_model(len(input2idx), len(target2idx))
    model.summary()

    cb = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
    model.fit(
        [enc_tr, dec_in_tr], dec_out_tr,
        validation_data=([enc_va, dec_in_va], dec_out_va),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=1
    )

    os.makedirs('saved_models', exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\n✅ Saved → {MODEL_PATH}")

    preds   = [predict_word(model, r, input2idx, idx2target, target2idx) for r, _ in test_pairs]
    refs    = [d for _, d in test_pairs]
    metrics = compute_metrics(preds, refs)
    print("\n📊 Model 8 — CNN+LSTM Results:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")
    return model, input2idx, idx2target, target2idx, metrics


if __name__ == '__main__':
    train()
