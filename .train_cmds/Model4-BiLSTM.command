#!/usr/bin/env bash
echo "══════════════════════════════════════"
echo "  Training: Model4-BiLSTM"
echo "══════════════════════════════════════"
cd "/Users/HetviSheth/Transliterator"
python3 "/Users/HetviSheth/Transliterator/models/model4_bilstm.py" 2>&1 | tee "/Users/HetviSheth/Transliterator/logs/model4_bilstm.py.log"
echo ""
echo "✅ Model4-BiLSTM — DONE  |  Log: /Users/HetviSheth/Transliterator/logs/model4_bilstm.py.log"
echo "Press any key to close..."
read -n 1
