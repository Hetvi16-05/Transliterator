#!/usr/bin/env bash
echo "══════════════════════════════════════"
echo "  Training: Model3-LSTM"
echo "══════════════════════════════════════"
cd "/Users/HetviSheth/Transliterator"
python3 "/Users/HetviSheth/Transliterator/models/model3_lstm.py" 2>&1 | tee "/Users/HetviSheth/Transliterator/logs/model3_lstm.py.log"
echo ""
echo "✅ Model3-LSTM — DONE  |  Log: /Users/HetviSheth/Transliterator/logs/model3_lstm.py.log"
echo "Press any key to close..."
read -n 1
