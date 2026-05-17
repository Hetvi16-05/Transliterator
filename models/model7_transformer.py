"""
model7_transformer.py
----------------------
Model 7: Transformer (built entirely from scratch)
- No pretrained weights, no HuggingFace
- Multi-Head Self-Attention + Positional Encoding
- Same architecture as "Attention Is All You Need" (Vaswani 2017)

Architecture for viva explanation:
  1. Positional Encoding  → gives order info to token embeddings
  2. Multi-Head Attention → attends to different positions in parallel
  3. Feed-Forward Network → position-wise dense layers
  4. Layer Normalization  → stabilises training
  5. Encoder: stack of N encoder blocks
  6. Decoder: stack of N decoder blocks (with masked self-attention)

Person 4 is responsible for this file.

Usage:
    python models/model7_transformer.py
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
D_MODEL     = 64    # embedding dimension
NUM_HEADS   = 4     # attention heads (D_MODEL must be divisible by NUM_HEADS)
DFF         = 128   # feed-forward inner dimension
NUM_LAYERS  = 2     # encoder & decoder stack depth
DROPOUT     = 0.1
BATCH_SIZE  = 16
EPOCHS      = 100
MAX_ENC_LEN = 30
MAX_DEC_LEN = 30
MODEL_PATH  = 'saved_models/model7_transformer.keras'


# ──────────────────────────────────────────────
# BUILDING BLOCKS
# ──────────────────────────────────────────────

def positional_encoding(max_len, d_model):
    """Sine/cosine positional encoding (Vaswani 2017)."""
    positions = np.arange(max_len)[:, np.newaxis]        # (max_len, 1)
    dims      = np.arange(d_model)[np.newaxis, :]        # (1, d_model)
    angles    = positions / np.power(10000, (2*(dims//2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)  # (1, max_len, d_model)


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.depth     = d_model // num_heads
        self.d_model   = d_model
        self.Wq = layers.Dense(d_model)
        self.Wk = layers.Dense(d_model)
        self.Wv = layers.Dense(d_model)
        self.Wo = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, [0, 2, 1, 3])

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.split_heads(self.Wq(q), batch_size)
        k = self.split_heads(self.Wk(k), batch_size)
        v = self.split_heads(self.Wv(v), batch_size)

        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.depth, tf.float32))
        if mask is not None:
            matmul_qk += (mask * -1e9)
        weights = tf.nn.softmax(matmul_qk, axis=-1)
        out     = tf.matmul(weights, v)

        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, -1, self.d_model))
        return self.Wo(out)


class EncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout, **kwargs):
        super().__init__(**kwargs)
        self.mha  = MultiHeadAttention(d_model, num_heads)
        self.ffn1 = layers.Dense(dff, activation='relu')
        self.ffn2 = layers.Dense(d_model)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False, mask=None):
        attn = self.mha(x, x, x, mask)
        attn = self.drop1(attn, training=training)
        x    = self.norm1(x + attn)
        ffn  = self.ffn2(self.ffn1(x))
        ffn  = self.drop2(ffn, training=training)
        return self.norm2(x + ffn)


class DecoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout, **kwargs):
        super().__init__(**kwargs)
        self.mha1  = MultiHeadAttention(d_model, num_heads)   # masked self-attention
        self.mha2  = MultiHeadAttention(d_model, num_heads)   # cross-attention
        self.ffn1  = layers.Dense(dff, activation='relu')
        self.ffn2  = layers.Dense(d_model)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)
        self.drop3 = layers.Dropout(dropout)

    def call(self, x, enc_out, training=False, look_ahead_mask=None, padding_mask=None):
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.drop1(attn1, training=training)
        x     = self.norm1(x + attn1)

        attn2 = self.mha2(x, enc_out, enc_out, padding_mask)
        attn2 = self.drop2(attn2, training=training)
        x     = self.norm2(x + attn2)

        ffn   = self.ffn2(self.ffn1(x))
        ffn   = self.drop3(ffn, training=training)
        return self.norm3(x + ffn)


# ──────────────────────────────────────────────
# FULL TRANSFORMER MODEL
# ──────────────────────────────────────────────

def build_transformer(input_vocab, target_vocab):
    pos_enc_enc = positional_encoding(MAX_ENC_LEN, D_MODEL)
    pos_enc_dec = positional_encoding(MAX_DEC_LEN, D_MODEL)

    enc_input = layers.Input(shape=(MAX_ENC_LEN,), name='enc_input')
    dec_input = layers.Input(shape=(MAX_DEC_LEN,), name='dec_input')

    # Encoder
    x = layers.Embedding(input_vocab, D_MODEL)(enc_input)
    x = x + pos_enc_enc
    x = layers.Dropout(DROPOUT)(x)
    for i in range(NUM_LAYERS):
        x = EncoderBlock(D_MODEL, NUM_HEADS, DFF, DROPOUT, name=f'enc_block_{i}')(x)
    enc_out = x

    # Decoder
    y = layers.Embedding(target_vocab, D_MODEL)(dec_input)
    y = y + pos_enc_dec
    y = layers.Dropout(DROPOUT)(y)
    for i in range(NUM_LAYERS):
        y = DecoderBlock(D_MODEL, NUM_HEADS, DFF, DROPOUT, name=f'dec_block_{i}')(y, enc_out)

    outputs = layers.Dense(target_vocab, activation='softmax')(y)

    model = Model([enc_input, dec_input], outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ──────────────────────────────────────────────
# DATA + TRAIN
# ──────────────────────────────────────────────

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
    print("\n🚀 Model 7: Transformer (from scratch) — Training...\n")
    (enc_tr, dec_in_tr, dec_out_tr,
     enc_va, dec_in_va, dec_out_va,
     input2idx, idx2input, target2idx, idx2target,
     train_pairs, test_pairs) = load_data()

    model = build_transformer(len(input2idx), len(target2idx))
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
    print("\n📊 Model 7 — Transformer Results:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")
    return model, input2idx, idx2target, target2idx, metrics


if __name__ == '__main__':
    train()
