"""
utils.py
--------
Shared utilities: vocab building, data encoding, padding, metrics.
All 8 models import from here.
"""

import numpy as np
import pandas as pd
from jiwer import wer, cer


# ──────────────────────────────────────────────
# VOCABULARY
# ──────────────────────────────────────────────

def build_vocab(pairs):
    input_chars  = sorted(set(c for r, _ in pairs for c in r))
    target_chars = sorted(set(c for _, d in pairs for c in d))

    input_tokens  = ['<PAD>', '<SOS>', '<EOS>'] + input_chars
    target_tokens = ['<PAD>', '<SOS>', '<EOS>'] + target_chars

    input2idx  = {c: i for i, c in enumerate(input_tokens)}
    idx2input  = {i: c for i, c in enumerate(input_tokens)}
    target2idx = {c: i for i, c in enumerate(target_tokens)}
    idx2target = {i: c for i, c in enumerate(target_tokens)}

    return input2idx, idx2input, target2idx, idx2target


# ──────────────────────────────────────────────
# ENCODING
# ──────────────────────────────────────────────

def encode_sequences(pairs, input2idx, target2idx, max_enc_len=30, max_dec_len=30):
    """Return encoder_input, decoder_input, decoder_target as numpy arrays."""
    enc_inputs, dec_inputs, dec_targets = [], [], []

    for roman, devanagari in pairs:
        enc = [input2idx.get(c, 0) for c in roman]
        dec_in  = [target2idx['<SOS>']] + [target2idx.get(c, 0) for c in devanagari]
        dec_out = [target2idx.get(c, 0) for c in devanagari] + [target2idx['<EOS>']]

        enc     = pad_seq(enc,     max_enc_len)
        dec_in  = pad_seq(dec_in,  max_dec_len)
        dec_out = pad_seq(dec_out, max_dec_len)

        enc_inputs.append(enc)
        dec_inputs.append(dec_in)
        dec_targets.append(dec_out)

    return (np.array(enc_inputs, dtype=np.int32),
            np.array(dec_inputs, dtype=np.int32),
            np.array(dec_targets, dtype=np.int32))


def pad_seq(seq, max_len):
    return seq[:max_len] + [0] * max(0, max_len - len(seq))


# ──────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────

def compute_metrics(predictions, references):
    """Compute WER, CER, and word accuracy."""
    word_acc = np.mean([p == r for p, r in zip(predictions, references)])
    w = wer(references, predictions)
    c = cer(references, predictions)
    return {
        'WER': round(w, 4),
        'CER': round(c, 4),
        'Word_Accuracy': round(word_acc, 4)
    }


def load_pairs(csv_path):
    df = pd.read_csv(csv_path)
    return list(zip(df['roman'], df['devanagari']))
