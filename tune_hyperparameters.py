import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.utils import build_vocab, pad_seq, configure_gpu

# Load the newly combined dataset
def load_combined_data(filepath, max_samples=None):
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                pairs.append((parts[1], parts[0])) # Roman, Devanagari
    if max_samples:
        pairs = pairs[:max_samples]
    return pairs

MAX_ENC_LEN = 30
MAX_DEC_LEN = 30
BATCH_SIZE = 128
EPOCHS = 8 # Enough to see which converges better without taking 10 hours

def build_gru(input_vocab_size, target_vocab_size, emb_dim, units):
    enc_inputs = layers.Input(shape=(MAX_ENC_LEN,))
    enc_emb = layers.Embedding(input_vocab_size, emb_dim)(enc_inputs)
    enc_out, enc_state = layers.GRU(units, return_state=True)(enc_emb)
    
    dec_inputs = layers.Input(shape=(MAX_DEC_LEN,))
    dec_emb = layers.Embedding(target_vocab_size, emb_dim)(dec_inputs)
    dec_gru = layers.GRU(units, return_sequences=True, return_state=True)
    dec_out, _ = dec_gru(dec_emb, initial_state=enc_state)
    
    outputs = layers.Dense(target_vocab_size, activation='softmax')(dec_out)
    return Model([enc_inputs, dec_inputs], outputs)

class BahdanauAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V  = layers.Dense(1)
        self._units = units

    def get_config(self):
        config = super().get_config()
        config.update({"units": self._units})
        return config

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

def build_attention(input_vocab_size, target_vocab_size, emb_dim, units):
    attention = BahdanauAttention(units)
    enc_inputs = layers.Input(shape=(MAX_ENC_LEN,))
    enc_emb    = layers.Embedding(input_vocab_size, emb_dim)(enc_inputs)
    enc_out, enc_h, enc_c = layers.LSTM(units, return_sequences=True, return_state=True)(enc_emb)
    
    dec_inputs = layers.Input(shape=(MAX_DEC_LEN,))
    dec_emb    = layers.Embedding(target_vocab_size, emb_dim)(dec_inputs)
    
    context, _ = attention(enc_h, enc_out)
    context_exp = layers.Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, MAX_DEC_LEN, 1]))(context)
    
    dec_lstm_in = layers.Concatenate(axis=-1)([dec_emb, context_exp])
    dec_out, _, _ = layers.LSTM(units, return_sequences=True, return_state=True)(
                                  dec_lstm_in, initial_state=[enc_h, enc_c])
    outputs = layers.Dense(target_vocab_size, activation='softmax')(dec_out)
    return Model([enc_inputs, dec_inputs], outputs)

def tune_models():
    configure_gpu()
    print("Loading Combined Dataset...")
    all_pairs = load_combined_data("dataset/custom_combined_dataset.tsv")
    print(f"Total pairs: {len(all_pairs)}")
    
    train_pairs, test_pairs = train_test_split(all_pairs, test_size=0.1, random_state=42)
    train_pairs, val_pairs = train_test_split(train_pairs, test_size=0.1, random_state=42)
    
    input2idx, idx2input, target2idx, idx2target = build_vocab(all_pairs)
    
    def prepare_data(pairs):
        enc_data, dec_in_data, dec_tgt_data = [], [], []
        for ro, hi in pairs:
            enc = [input2idx.get(c, 0) for c in ro.lower()]
            dec = [target2idx['<SOS>']] + [target2idx.get(c, 0) for c in hi] + [target2idx['<EOS>']]
            enc_data.append(pad_seq(enc, MAX_ENC_LEN))
            dec_in_data.append(pad_seq(dec[:-1], MAX_DEC_LEN))
            dec_tgt_data.append(pad_seq(dec[1:], MAX_DEC_LEN))
        return np.array(enc_data), np.array(dec_in_data), np.array(dec_tgt_data)

    X_train_enc, X_train_dec, y_train = prepare_data(train_pairs)
    X_val_enc, X_val_dec, y_val = prepare_data(val_pairs)
    
    configs = [
        {'emb_dim': 128, 'units': 128, 'lr': 0.001},
        {'emb_dim': 256, 'units': 256, 'lr': 0.001},
        {'emb_dim': 128, 'units': 256, 'lr': 0.0005},
    ]
    
    results = []
    
    for m_type, builder in [('GRU', build_gru), ('Attention', build_attention)]:
        for i, cfg in enumerate(configs):
            print(f"\n[{m_type} Config {i+1}/{len(configs)}] Dim={cfg['emb_dim']}, Units={cfg['units']}, LR={cfg['lr']}")
            
            model = builder(len(input2idx), len(target2idx), cfg['emb_dim'], cfg['units'])
            model.compile(optimizer=tf.keras.optimizers.Adam(cfg['lr']),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            
            history = model.fit(
                [X_train_enc, X_train_dec], y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=([X_val_enc, X_val_dec], y_val),
                verbose=1
            )
            
            train_acc = history.history['accuracy'][-1]
            val_acc = history.history['val_accuracy'][-1]
            
            results.append({
                'Model': m_type,
                'Embedding Dim': cfg['emb_dim'],
                'Units': cfg['units'],
                'Learning Rate': cfg['lr'],
                'Train Acc': f"{train_acc*100:.2f}%",
                'Val Acc': f"{val_acc*100:.2f}%"
            })
            
    df = pd.DataFrame(results)
    df.to_csv("evaluation/hyperparameter_tuning.csv", index=False)
    print("\n✅ Tuning Complete. Results saved to evaluation/hyperparameter_tuning.csv")
    print(df.to_string(index=False))

if __name__ == "__main__":
    tune_models()
