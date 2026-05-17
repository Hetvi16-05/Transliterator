#!/usr/bin/env bash
echo "══════════════════════════════════════════════"
echo "  Model 6 — LSTM + Attention (Metal GPU)"
echo "══════════════════════════════════════════════"
cd "/Users/HetviSheth/Transliterator"
/Applications/miniconda3/envs/ml/bin/python models/model6_attention.py 2>&1 | tee "/Users/HetviSheth/Transliterator/logs/model6_attention_metal.log"
echo ""
echo "✅ Model 6 Attention — DONE  |  Log: /Users/HetviSheth/Transliterator/logs/model6_attention_metal.log"
echo "Press any key to close..."
read -n 1
