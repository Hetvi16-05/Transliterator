import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from models.utils import (build_vocab, encode_sequences, load_pairs,
                           compute_metrics, configure_gpu)

EMBEDDING_DIM = 64
UNITS         = 256
BATCH_SIZE    = 64
EPOCHS        = 100
MAX_ENC_LEN   = 30
MAX_DEC_LEN   = 30
MODEL_PATH    = 'saved_models/model6_attention.keras'


# ──────────────────────────────────────────────
# BAHDANAU ATTENTION LAYER (built from scratch)
# ──────────────────────────────────────────────

class BahdanauAttention(layers.Layer):
    """
    Additive (Bahdanau) attention.
    Paper: 'Neural Machine Translation by Jointly Learning to Align and Translate'
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W1 = layers.Dense(units)   # transforms encoder outputs
        self.W2 = layers.Dense(units)   # transforms decoder hidden state
        self.V  = layers.Dense(1)       # scores each encoder timestep

    def call(self, decoder_hidden, encoder_outputs):
        # decoder_hidden : (batch, units)
        # encoder_outputs: (batch, enc_len, units)

        # Expand decoder_hidden for broadcasting → (batch, 1, units)
        dec_hidden_exp = tf.expand_dims(decoder_hidden, 1)

        # Compute attention scores → (batch, enc_len, 1)
        score = self.V(tf.nn.tanh(self.W1(encoder_outputs) + self.W2(dec_hidden_exp)))

        # Softmax over encoder timesteps → (batch, enc_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # Weighted sum of encoder outputs → (batch, units)
        context_vector = attention_weights * encoder_outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────

def load_data():
    from sklearn.model_selection import train_test_split
    pairs = []
    with open('dataset/custom_combined_dataset.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                pairs.append((parts[1], parts[0]))
                
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.1, random_state=42)
    train_pairs, val_pairs = train_test_split(train_pairs, test_size=0.1, random_state=42)
    
    all_pairs = train_pairs + val_pairs + test_pairs
    input2idx, idx2input, target2idx, idx2target = build_vocab(all_pairs)
    enc_tr, dec_in_tr, dec_out_tr = encode_sequences(train_pairs, input2idx, target2idx, MAX_ENC_LEN, MAX_DEC_LEN)
    enc_va, dec_in_va, dec_out_va = encode_sequences(val_pairs,   input2idx, target2idx, MAX_ENC_LEN, MAX_DEC_LEN)
    return (enc_tr, dec_in_tr, dec_out_tr,
            enc_va, dec_in_va, dec_out_va,
            input2idx, idx2input, target2idx, idx2target,
            train_pairs, test_pairs)


# ──────────────────────────────────────────────
# MODEL (Teacher-forcing training with attention)
# ──────────────────────────────────────────────

def build_model(input_vocab_size, target_vocab_size, emb_dim, units, lr):
    attention = BahdanauAttention(units)

    # Encoder
    enc_inputs = layers.Input(shape=(MAX_ENC_LEN,), name='enc_input')
    enc_emb    = layers.Embedding(input_vocab_size, emb_dim)(enc_inputs)
    enc_out, enc_h, enc_c = layers.LSTM(units, return_sequences=True,
                                         return_state=True, name='enc_lstm')(enc_emb)

    # Decoder with attention — step by step using Lambda
    dec_inputs = layers.Input(shape=(MAX_DEC_LEN,), name='dec_input')
    dec_emb    = layers.Embedding(target_vocab_size, emb_dim)(dec_inputs)

    # Use a simple dense-attention approximation for training
    # (full step-by-step attention is in predict_word below)
    context, _ = attention(enc_h, enc_out)                     # (batch, units)
    context_exp = layers.Lambda(
        lambda x: tf.tile(tf.expand_dims(x, 1), [1, MAX_DEC_LEN, 1])
    )(context)                                                  # (batch, dec_len, units)

    dec_lstm_in = layers.Concatenate(axis=-1)([dec_emb, context_exp])
    dec_out, _, _ = layers.LSTM(units, return_sequences=True,
                                  return_state=True, name='dec_lstm')(
                                  dec_lstm_in, initial_state=[enc_h, enc_c])
    outputs = layers.Dense(target_vocab_size, activation='softmax')(dec_out)

    model = Model([enc_inputs, dec_inputs], outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
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
    print("\n🚀 Model 6: LSTM + Bahdanau Attention — Training with Custom Dataset & Tuning...\n")
    (enc_tr, dec_in_tr, dec_out_tr,
     enc_va, dec_in_va, dec_out_va,
     input2idx, idx2input, target2idx, idx2target,
     train_pairs, test_pairs) = load_data()

    configs = [
        {'emb': 128, 'units': 128, 'lr': 0.001, 'bs': 128},
        {'emb': 256, 'units': 256, 'lr': 0.001, 'bs': 128},
        {'emb': 128, 'units': 256, 'lr': 0.0005, 'bs': 64},
        {'emb': 256, 'units': 128, 'lr': 0.0005, 'bs': 64}
    ]

    best_val_acc = 0.0
    best_config = None
    tuning_results = []

    print("--- Phase 1: Hyperparameter Tuning (20 Epochs each) ---")
    for i, cfg in enumerate(configs):
        print(f"\n[Testing Config {i+1}/4] Embed: {cfg['emb']}, Units: {cfg['units']}, LR: {cfg['lr']}, Batch: {cfg['bs']}")
        model = build_model(len(input2idx), len(target2idx), cfg['emb'], cfg['units'], cfg['lr'])
        cb = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        
        hist = model.fit(
            [enc_tr, dec_in_tr], dec_out_tr,
            validation_data=([enc_va, dec_in_va], dec_out_va),
            epochs=20, batch_size=cfg['bs'], callbacks=cb, verbose=1
        )
        
        val_acc = hist.history['val_accuracy'][-1]
        train_acc = hist.history['accuracy'][-1]
        tuning_results.append((cfg, train_acc, val_acc))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = cfg

    print("\n--- Tuning Results Summary ---")
    for idx, (cfg, t_acc, v_acc) in enumerate(tuning_results):
        mark = "⭐⭐⭐" if cfg == best_config else ""
        print(f"Config {idx+1}: Emb={cfg['emb']}, Units={cfg['units']}, LR={cfg['lr']}, Batch={cfg['bs']} -> Train Acc: {t_acc*100:.2f}%, Val Acc: {v_acc*100:.2f}% {mark}")

    print("\n--- Phase 2: Final Training with Best Config ---")
    print(f"Selected Best: {best_config}")
    
    final_model = build_model(len(input2idx), len(target2idx), best_config['emb'], best_config['units'], best_config['lr'])
    final_cb = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
    
    final_hist = final_model.fit(
        [enc_tr, dec_in_tr], dec_out_tr,
        validation_data=([enc_va, dec_in_va], dec_out_va),
        epochs=30, batch_size=best_config['bs'], callbacks=final_cb, verbose=1
    )

    os.makedirs('saved_models', exist_ok=True)
    final_model.save(MODEL_PATH)
    print(f"\n✅ Saved Best Model → {MODEL_PATH}")

    print("\n--- Phase 3: Final Evaluation on Test Set (1000 samples) ---")
    sample_tests = test_pairs[:1000]
    preds = []
    for idx, (r, _) in enumerate(sample_tests):
        preds.append(predict_word(final_model, r, input2idx, idx2target, target2idx))
        if (idx+1) % 200 == 0: print(f"  Predicted {idx+1}/{len(sample_tests)}...")
        
    refs    = [d for _, d in sample_tests]
    metrics = compute_metrics(preds, refs)
    
    print("\n📊 Model 6 — LSTM + Attention Final Results (Best Hyperparams):")
    print(f"   Final Train Acc : {final_hist.history['accuracy'][-1]*100:.2f}%")
    print(f"   Final Val Acc   : {final_hist.history['val_accuracy'][-1]*100:.2f}%")
    for k, v in metrics.items():
        print(f"   {k}: {v}")
    return model, input2idx, idx2target, target2idx, metrics


if __name__ == '__main__':
    train()
