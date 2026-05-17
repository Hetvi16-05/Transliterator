"""
utils.py
--------
Shared utilities: vocab building, data encoding, padding, metrics.
All 8 models import from here.

Dataset: Dakshina v1.0 — Hindi lexicons (TSV format)
  Column 0: Devanagari (target)
  Column 1: Roman / transliterated (input)
  Column 2: frequency count (ignored)
"""

import os
import numpy as np
import pandas as pd
from jiwer import wer, cer
import tensorflow as tf

# ──────────────────────────────────────────────
# DATASET PATHS  (absolute, based on this file's location)
# ──────────────────────────────────────────────

_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_DIR = os.path.join(_REPO_ROOT, 'dakshina_dataset_v1.0', 'hi', 'lexicons')

TRAIN_TSV = os.path.join(_DATASET_DIR, 'hi.translit.sampled.train.tsv')
VAL_TSV   = os.path.join(_DATASET_DIR, 'hi.translit.sampled.dev.tsv')
TEST_TSV  = os.path.join(_DATASET_DIR, 'hi.translit.sampled.test.tsv')


# ──────────────────────────────────────────────
# GPU CONFIGURATION
# ──────────────────────────────────────────────

def configure_gpu():
    """Enable GPU memory growth to avoid OOM errors."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅  GPU configured: {[g.name for g in gpus]}")
        except RuntimeError as e:
            print(f"⚠️  GPU config error: {e}")
    else:
        print("ℹ️  No GPU found — running on CPU.")


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_pairs(tsv_path=None, split='train'):
    """
    Load (roman, devanagari) pairs from a Dakshina TSV file.

    Parameters
    ----------
    tsv_path : str | None
        Explicit path to a TSV file.  If None, ``split`` is used to
        select the canonical train / dev / test file.
    split : str
        One of 'train', 'val' / 'dev', 'test'.  Ignored when
        ``tsv_path`` is supplied explicitly.

    Returns
    -------
    list[tuple[str, str]]
        List of (roman_input, devanagari_target) pairs.
    """
    if tsv_path is None:
        mapping = {
            'train': TRAIN_TSV,
            'val':   VAL_TSV,
            'dev':   VAL_TSV,
            'test':  TEST_TSV,
        }
        tsv_path = mapping[split]

    df = pd.read_csv(
        tsv_path,
        sep='\t',
        header=None,
        names=['devanagari', 'roman', 'freq'],
        encoding='utf-8',
        on_bad_lines='skip',
    )
    # Drop rows with missing values in the columns we need
    df = df.dropna(subset=['roman', 'devanagari'])
    df['roman']      = df['roman'].astype(str).str.strip().str.lower()
    df['devanagari'] = df['devanagari'].astype(str).str.strip()

    # Filter out empty strings
    df = df[(df['roman'] != '') & (df['devanagari'] != '')]

    pairs = list(zip(df['roman'], df['devanagari']))
    return pairs


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

    return (np.array(enc_inputs,  dtype=np.int32),
            np.array(dec_inputs,  dtype=np.int32),
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
