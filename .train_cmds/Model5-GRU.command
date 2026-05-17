#!/usr/bin/env bash
echo "══════════════════════════════════════"
echo "  Training: Model5-GRU"
echo "══════════════════════════════════════"
cd "/Users/HetviSheth/Transliterator"
python3 "/Users/HetviSheth/Transliterator/models/model5_gru.py" 2>&1 | tee "/Users/HetviSheth/Transliterator/logs/model5_gru.py.log"
echo ""
echo "✅ Model5-GRU — DONE  |  Log: /Users/HetviSheth/Transliterator/logs/model5_gru.py.log"
echo "Press any key to close..."
read -n 1
