import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TransliterationDataset, collate_fn, Vocabulary
from model import Encoder, Attention, Decoder, Seq2Seq
from train import evaluate
from utils import load_json

def main():
    parser = argparse.ArgumentParser(description="Evaluate the Transliteration Model")
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt', help='Path to model weights')
    parser.add_argument('--config', type=str, default='checkpoints/config.json', help='Path to config file')
    parser.add_argument('--src_vocab', type=str, default='checkpoints/source_vocab.json', help='Path to source vocab')
    parser.add_argument('--tgt_vocab', type=str, default='checkpoints/target_vocab.json', help='Path to target vocab')
    parser.add_argument('--test_data', type=str, default='hin_test.json', help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cpu or cuda)')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return

    device = torch.device(args.device)
    config = load_json(args.config)
    
    print("Loading vocabularies...")
    src_vocab = Vocabulary.from_dict(load_json(args.src_vocab))
    tgt_vocab = Vocabulary.from_dict(load_json(args.tgt_vocab))

    print(f"Loading dataset from {args.test_data}...")
    test_dataset = TransliterationDataset(
        args.test_data, 
        src_vocab=src_vocab, 
        tgt_vocab=tgt_vocab, 
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )

    print("Initializing model...")
    enc = Encoder(len(src_vocab), config['EMBEDDING_DIM'], config['HIDDEN_DIM'], config['HIDDEN_DIM'], config['ENC_DROPOUT'])
    attn = Attention(config['HIDDEN_DIM'], config['HIDDEN_DIM'])
    dec = Decoder(len(tgt_vocab), config['EMBEDDING_DIM'], config['HIDDEN_DIM'], config['HIDDEN_DIM'], config['DEC_DROPOUT'], attn)
    
    model = Seq2Seq(enc, dec, src_vocab.char2idx['<pad>'], device).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.char2idx['<pad>'])
    
    print(f"Evaluating model on {device}...")
    loss, metrics = evaluate(model, test_loader, criterion, device, tgt_vocab)
    
    print("\n" + "=" * 40)
    print("           EVALUATION RESULTS           ")
    print("=" * 40)
    print(f"Test Loss:           {loss:.4f}")
    print(f"Character Error Rate (CER): {metrics['cer']:.4f}")
    print(f"Word Error Rate (WER):      {metrics['wer']:.4f}")
    print(f"Character Accuracy:         {metrics['char_accuracy'] * 100:.2f}%")
    print(f"Word Accuracy:              {metrics['word_accuracy'] * 100:.2f}%")
    print(f"BLEU Score:                 {metrics['bleu'] * 100:.2f}")
    print("=" * 40)

if __name__ == "__main__":
    main()
