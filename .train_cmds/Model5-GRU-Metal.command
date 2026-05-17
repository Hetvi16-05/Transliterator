#!/usr/bin/env bash
echo "══════════════════════════════════════════════"
echo "  Model 5 — GRU (Metal GPU / Apple Silicon)"
echo "══════════════════════════════════════════════"
cd "/Users/HetviSheth/Transliterator"
/Applications/miniconda3/envs/ml/bin/python models/model5_gru.py 2>&1 | tee "/Users/HetviSheth/Transliterator/logs/model5_gru_metal.log"
echo ""
echo "✅ Model 5 GRU — DONE  |  Log: /Users/HetviSheth/Transliterator/logs/model5_gru_metal.log"
echo "Press any key to close..."
read -n 1
