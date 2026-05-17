#!/usr/bin/env bash
echo "══════════════════════════════════════"
echo "  Training: Model2-RNN"
echo "══════════════════════════════════════"
cd "/Users/HetviSheth/Transliterator"
python3 "/Users/HetviSheth/Transliterator/models/model2_rnn.py" 2>&1 | tee "/Users/HetviSheth/Transliterator/logs/model2_rnn.py.log"
echo ""
echo "✅ Model2-RNN — DONE  |  Log: /Users/HetviSheth/Transliterator/logs/model2_rnn.py.log"
echo "Press any key to close..."
read -n 1
