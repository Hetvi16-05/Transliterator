"""
model1_rulebased.py
--------------------
Model 1: Rule-Based Character Mapping
- No neural network, no training required
- Uses a hand-crafted mapping table of Roman → Devanagari phonemes
- Serves as the BASELINE to beat

Person 1 is responsible for this file.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.utils import load_pairs, compute_metrics

# ──────────────────────────────────────────────────────────
# FULL PHONEME MAPPING TABLE
# Ordered longest-match first so "sh" is tried before "s"
# ──────────────────────────────────────────────────────────

PHONEME_MAP = [
    # Multi-char mappings (MUST come before single-char)
    ("ksh", "क्ष"), ("gya", "ज्ञ"), ("tra", "त्र"),
    ("shri", "श्री"), ("shr", "श्र"), ("str", "स्त्र"),
    ("kkh", "क्ख"), ("ddh", "द्ध"), ("ndh", "न्ध"),
    ("ndr", "न्द्र"), ("mbh", "म्भ"), ("nth", "न्थ"),
    ("rdh", "र्ध"), ("rth", "र्थ"), ("rsh", "र्ष"),
    ("aa", "आ"), ("ii", "ई"), ("uu", "ऊ"),
    ("ee", "ई"), ("oo", "ऊ"), ("ai", "ऐ"),
    ("au", "औ"), ("ou", "औ"), ("ae", "ऐ"),
    ("ri", "ऋ"), ("ru", "रु"),
    ("sh", "श"), ("ch", "च"), ("th", "थ"),
    ("ph", "फ"), ("bh", "भ"), ("gh", "घ"),
    ("dh", "ध"), ("jh", "झ"), ("kh", "ख"),
    ("nh", "न्ह"), ("mh", "म्ह"), ("rh", "र्ह"),
    ("ng", "ङ"), ("nk", "ङ्क"), ("nc", "ञ्च"),
    ("nd", "न्द"), ("nt", "न्त"), ("mb", "म्ब"),
    ("mp", "म्प"), ("lt", "ल्त"), ("lp", "ल्प"),
    ("rn", "र्न"), ("rm", "र्म"), ("rv", "र्व"),
    ("tt", "ट्ट"), ("dd", "ड्ड"), ("nn", "न्न"),
    ("ll", "ल्ल"), ("ss", "स्स"), ("mm", "म्म"),
    ("pp", "प्प"), ("bb", "ब्ब"), ("kk", "क्क"),
    ("ck", "क्क"), ("qu", "क्व"),

    # Single-char vowels
    ("a", "अ"), ("e", "ए"), ("i", "इ"),
    ("o", "ओ"), ("u", "उ"),

    # Single-char consonants
    ("k", "क"), ("g", "ग"), ("c", "क"),
    ("j", "ज"), ("t", "त"), ("d", "द"),
    ("n", "न"), ("p", "प"), ("b", "ब"),
    ("m", "म"), ("y", "य"), ("r", "र"),
    ("l", "ल"), ("v", "व"), ("w", "व"),
    ("s", "स"), ("h", "ह"), ("z", "ज़"),
    ("f", "फ"), ("q", "क"), ("x", "क्स"),

    # Digits and punctuation pass-through
    ("0", "०"), ("1", "१"), ("2", "२"),
    ("3", "३"), ("4", "४"), ("5", "५"),
    ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९"),
]


def transliterate(text: str) -> str:
    """Convert a Roman-script Hindi word/sentence to Devanagari."""
    text  = text.lower().strip()
    result = ""
    i = 0
    while i < len(text):
        matched = False
        # Try longest match first
        for roman, devanagari in PHONEME_MAP:
            if text[i:i+len(roman)] == roman:
                result  += devanagari
                i       += len(roman)
                matched  = True
                break
        if not matched:
            result += text[i]   # pass through unknown chars (space, etc.)
            i += 1
    return result


def transliterate_sentence(sentence: str) -> str:
    """Transliterate word by word to preserve spaces."""
    words = sentence.split()
    return " ".join(transliterate(w) for w in words)


# ──────────────────────────────────────────────────────────
# EVALUATE
# ──────────────────────────────────────────────────────────

def evaluate(test_path='dataset/processed/test.csv'):
    pairs = load_pairs(test_path)
    preds = [transliterate(r) for r, _ in pairs]
    refs  = [d for _, d in pairs]
    metrics = compute_metrics(preds, refs)
    print("\n📊 Model 1 — Rule-Based Results:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")
    return metrics


if __name__ == '__main__':
    # Quick demo
    tests = [
        "narendra modi bharat ke pradhanmantri hain",
        "dilli mein india gate sthit hai",
        "aaj mausam accha hai",
    ]
    print("🔤 Model 1: Rule-Based Transliterator\n")
    for t in tests:
        print(f"  IN : {t}")
        print(f"  OUT: {transliterate_sentence(t)}\n")

    evaluate()
