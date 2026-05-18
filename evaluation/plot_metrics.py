import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.utils import load_pairs, build_vocab, pad_seq, compute_metrics

LOG_DIR = "../logs"
EVAL_DIR = "."

def parse_log(logfile):
    epochs = []
    accs = []
    losses = []
    val_accs = []
    val_losses = []
    
    if not os.path.exists(logfile):
        return None
        
    with open(logfile, 'r', encoding='utf-8') as f:
        for line in f:
            # We are looking for lines like:
            # 691/691 ... accuracy: 0.8885 - loss: 0.3782 - val_accuracy: 0.8589 - val_loss: 0.5381
            if "val_accuracy" in line and "val_loss" in line:
                try:
                    # Clean the line from terminal escape characters if present
                    line = re.sub(r'\x1b\[.*?m', '', line)
                    
                    acc_match = re.search(r'accuracy:\s+([0-9.]+)', line)
                    loss_match = re.search(r'- loss:\s+([0-9.]+)', line)
                    val_acc_match = re.search(r'val_accuracy:\s+([0-9.]+)', line)
                    val_loss_match = re.search(r'val_loss:\s+([0-9.]+)', line)
                    
                    if acc_match and loss_match and val_acc_match and val_loss_match:
                        epochs.append(len(epochs) + 1)
                        accs.append(float(acc_match.group(1)))
                        losses.append(float(loss_match.group(1)))
                        val_accs.append(float(val_acc_match.group(1)))
                        val_losses.append(float(val_loss_match.group(1)))
                except Exception:
                    pass
                    
    if not epochs:
        return None
    return {'epoch': epochs, 'accuracy': accs, 'loss': losses, 'val_accuracy': val_accs, 'val_loss': val_losses}

def plot_metrics():
    print("Generating Training Graphs...")
    logs = {
        "Model 5 (GRU)": os.path.join(LOG_DIR, "model5_gru_metal.log"),
        "Model 6 (Attention)": os.path.join(LOG_DIR, "model6_attention_metal.log")
    }
    
    plt.figure(figsize=(14, 6))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    for model_name, logfile in logs.items():
        data = parse_log(logfile)
        if data:
            plt.plot(data['epoch'], data['accuracy'], label=f'{model_name} Train Acc', linestyle='--')
            plt.plot(data['epoch'], data['val_accuracy'], label=f'{model_name} Val Acc')
    
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    for model_name, logfile in logs.items():
        data = parse_log(logfile)
        if data:
            plt.plot(data['epoch'], data['loss'], label=f'{model_name} Train Loss', linestyle='--')
            plt.plot(data['epoch'], data['val_loss'], label=f'{model_name} Val Loss')
            
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(EVAL_DIR, "training_metrics.png")
    plt.savefig(plot_path)
    print(f"✅ Graph saved to {plot_path}")

def evaluate_models():
    print("\nEvaluating Models on Test Set (Warning: This may take a few minutes)...")
    
    all_pairs = load_pairs(split='train') + load_pairs(split='val') + load_pairs(split='test')
    input2idx, idx2input, target2idx, idx2target = build_vocab(all_pairs)
    test_pairs = load_pairs(split='test')
    
    MAX_LEN = 30
    
    def evaluate_model(model_path, model_name, is_attention=False):
        if not os.path.exists(model_path):
            print(f"Skipping {model_name} (Not found at {model_path})")
            return None
            
        print(f"Loading {model_name}...")
        
        if is_attention:
            # Need to build model and load weights
            from models.model6_attention import build_model
            model = build_model(len(input2idx), len(target2idx))
            model.load_weights(model_path)
        else:
            model = tf.keras.models.load_model(model_path)
            
        print(f"Running predictions for {model_name}...")
        
        # We'll evaluate on 500 samples to save time, or use all 4502 if you prefer
        sample = test_pairs[:500] 
        preds = []
        refs = []
        
        for idx, (roman, target) in enumerate(sample):
            enc = np.array([pad_seq([input2idx.get(c,0) for c in roman.lower()], MAX_LEN)])
            dec = np.zeros((1, MAX_LEN), dtype=np.int32)
            dec[0,0] = target2idx['<SOS>']
            out = ''
            for i in range(1, MAX_LEN):
                p = model.predict([enc, dec], verbose=0)
                tid = int(np.argmax(p[0, i-1]))
                ch = idx2target.get(tid, '')
                if ch in ('<EOS>','<PAD>',''): break
                out += ch
                dec[0,i] = tid
            preds.append(out)
            refs.append(target)
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(sample)}...")
                
        metrics = compute_metrics(preds, refs)
        
        correct = sum(p == r for p, r in zip(preds, refs))
        word_acc = correct / len(sample)
        
        char_hits = []
        for p, r in zip(preds, refs):
            matches = sum(a == b for a, b in zip(p, r))
            char_hits.append(matches / max(len(r), 1))
        char_acc = np.mean(char_hits)
        
        return {
            'Model': model_name,
            'Word Accuracy': f"{word_acc*100:.2f}%",
            'Char Accuracy': f"{char_acc*100:.2f}%",
            'WER': f"{metrics['WER']:.4f}",
            'CER': f"{metrics['CER']:.4f}"
        }

    results = []
    
    res5 = evaluate_model("../saved_models/model5_gru.keras", "Model 5 (GRU)", is_attention=False)
    if res5: results.append(res5)
    
    res6 = evaluate_model("../saved_models/model6_attention.keras", "Model 6 (Attention)", is_attention=True)
    if res6: results.append(res6)
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv("results.csv", index=False)
        print("\n✅ Evaluation complete. Saved to results.csv")
        print(df.to_string(index=False))

if __name__ == "__main__":
    from models.utils import configure_gpu
    configure_gpu()
    plot_metrics()
    evaluate_models()
