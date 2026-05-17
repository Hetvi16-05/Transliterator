#!/usr/bin/env bash
echo "══════════════════════════════════════"
echo "  Training: Model8-CNN-LSTM"
echo "══════════════════════════════════════"
cd "/Users/HetviSheth/Transliterator"
python3 "/Users/HetviSheth/Transliterator/models/model8_cnn_lstm.py" 2>&1 | tee "/Users/HetviSheth/Transliterator/logs/model8_cnn_lstm.py.log"
echo ""
echo "✅ Model8-CNN-LSTM — DONE  |  Log: /Users/HetviSheth/Transliterator/logs/model8_cnn_lstm.py.log"
echo "Press any key to close..."
read -n 1
