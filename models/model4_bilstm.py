"""
model4_bilstm.py
-----------------
Model 4: Bidirectional LSTM Seq2Seq
- Encoder reads the Roman sequence BOTH forward and backward
- Gives the decoder richer context about each character's surroundings
- Better than plain LSTM for short sequences like words

Person 2 is responsible for this file.

Usage:
    python models/model4_bilstm.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from models.utils import build_vocab, encode_sequences, load_pairs, compute_metrics

EMBEDDING_DIM = 64
UNITS         = 256    # BiLSTM outputs 2*UNITS → project back to UNITS for decoder
BATCH_SIZE    = 16
EPOCHS        = 100
MAX_ENC_LEN   = 30
MAX_DEC_LEN   = 30
MODEL_PATH    = 'saved_models/model4_bilstm.keras'


def load_data():
    train_pairs = load_pairs('dataset/processed/train.csv')
    val_pairs   = load_pairs('dataset/processed/val.csv')
    test_pairs  = load_pairs('dataset/processed/test.csv')
    all_pairs   = train_pairs + val_pairs + test_pairs
    input2idx, idx2input, target2idx, idx2target = build_vocab(all_pairs)
    enc_tr, dec_in_tr, dec_out_tr = encode_sequences(train_pairs, input2idx, target2idx, MAX_ENC_LEN, MAX_DEC_LEN)
    enc_va, dec_in_va, dec_out_va = encode_sequences(val_pairs,   input2idx, target2idx, MAX_ENC_LEN, MAX_DEC_LEN)
    return (enc_tr, dec_in_tr, dec_out_tr,
            enc_va, dec_in_va, dec_out_va,
            input2idx, idx2input, target2idx, idx2target,
            train_pairs, test_pairs)


def build_model(input_vocab_size, target_vocab_size):
    # ── Encoder: Bidirectional LSTM ──
    encoder_inputs = layers.Input(shape=(MAX_ENC_LEN,), name='enc_input')
    enc_emb        = layers.Embedding(input_vocab_size, EMBEDDING_DIM)(encoder_inputs)

    # Bidirectional returns (output, fwd_h, fwd_c, bwd_h, bwd_c)
    bi_lstm        = layers.Bidirectional(
                        layers.LSTM(UNITS, return_state=True), name='bi_lstm')
    _, fwd_h, fwd_c, bwd_h, bwd_c = bi_lstm(enc_emb)

    # Merge forward + backward states
    enc_h = layers.Concatenate()([fwd_h, bwd_h])   # shape: (batch, 2*UNITS)
    enc_c = layers.Concatenate()([fwd_c, bwd_c])

    # Project down to UNITS for decoder compatibility
    enc_h = layers.Dense(UNITS, activation='tanh', name='proj_h')(enc_h)
    enc_c = layers.Dense(UNITS, activation='tanh', name='proj_c')(enc_c)

    # ── Decoder: Unidirectional LSTM ──
    decoder_inputs = layers.Input(shape=(MAX_DEC_LEN,), name='dec_input')
    dec_emb        = layers.Embedding(target_vocab_size, EMBEDDING_DIM)(decoder_inputs)
    dec_out, _, _  = layers.LSTM(UNITS, return_sequences=True,
                                  return_state=True, name='dec_lstm')(
                                  dec_emb, initial_state=[enc_h, enc_c])
    outputs        = layers.Dense(target_vocab_size, activation='softmax')(dec_out)

    model = Model([encoder_inputs, decoder_inputs], outputs)
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
    print("\n🚀 Model 4: Bidirectional LSTM — Training...\n")
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
    print("\n📊 Model 4 — BiLSTM Results:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")
    return model, input2idx, idx2target, target2idx, metrics


if __name__ == '__main__':
    train()
