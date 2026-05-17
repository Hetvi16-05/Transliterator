#!/usr/bin/env bash
echo "══════════════════════════════════════════════"
echo "  Model 2 — Simple RNN (Metal GPU)"
echo "══════════════════════════════════════════════"
cd "/Users/HetviSheth/Transliterator"
/Applications/miniconda3/envs/ml/bin/python models/model2_rnn.py 2>&1 | tee "/Users/HetviSheth/Transliterator/logs/model2_rnn_metal.log"
echo ""
echo "✅ Model 2 RNN — DONE  |  Log: $LOG"
echo "Press any key to close..."
read -n 1
