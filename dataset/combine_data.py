import os
import re
import pandas as pd

def clean_word(word):
    # Remove punctuation
    return re.sub(r'[^\w\s]', '', word).strip()

def process_sentence_file(filepath):
    pairs = set()
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return pairs
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                hi_sent, ro_sent = parts
                hi_words = [clean_word(w) for w in hi_sent.split()]
                ro_words = [clean_word(w).lower() for w in ro_sent.split()]
                
                # Only align if word counts match exactly
                if len(hi_words) == len(ro_words):
                    for h, r in zip(hi_words, ro_words):
                        if h and r and not re.search(r'[a-zA-Z]', h) and not re.search(r'[0-9]', h): # valid Hindi word
                            pairs.add((r, h))
    return pairs

def main():
    print("1. Extracting word pairs from sentence datasets...")
    file1 = "../dakshina_dataset_v1.0/hi/romanized/hi.romanized.rejoined.aligned.tsv"
    file2 = "../dakshina_dataset_v1.0/hi/romanized/hi.romanized.rejoined.aligned.cased_nopunct.tsv"
    
    pairs1 = process_sentence_file(file1)
    pairs2 = process_sentence_file(file2)
    
    sentence_pairs = pairs1.union(pairs2)
    print(f"Extracted {len(sentence_pairs)} unique word pairs from sentences.")
    
    print("2. Loading existing Lexicon datasets...")
    lexicon_dir = "../dakshina_dataset_v1.0/hi/lexicons"
    lex_pairs = set()
    for split in ['train', 'dev', 'test']:
        path = os.path.join(lexicon_dir, f"hi.translit.sampled.{split}.tsv")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        hi, ro = parts[0], parts[1]
                        lex_pairs.add((ro.lower(), hi))
                        
    print(f"Loaded {len(lex_pairs)} unique word pairs from Lexicons.")
    
    all_pairs = sentence_pairs.union(lex_pairs)
    print(f"Total Combined Unique Pairs: {len(all_pairs)}")
    
    # Save to a new combined TSV
    out_file = "../dataset/custom_combined_dataset.tsv"
    with open(out_file, 'w', encoding='utf-8') as f:
        for ro, hi in sorted(list(all_pairs)):
            f.write(f"{hi}\t{ro}\n")
            
    print(f"✅ Saved combined dataset to {out_file}")

if __name__ == "__main__":
    main()
