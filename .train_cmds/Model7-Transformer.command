#!/usr/bin/env bash
echo "══════════════════════════════════════"
echo "  Training: Model7-Transformer"
echo "══════════════════════════════════════"
cd "/Users/HetviSheth/Transliterator"
python3 "/Users/HetviSheth/Transliterator/models/model7_transformer.py" 2>&1 | tee "/Users/HetviSheth/Transliterator/logs/model7_transformer.py.log"
echo ""
echo "✅ Model7-Transformer — DONE  |  Log: /Users/HetviSheth/Transliterator/logs/model7_transformer.py.log"
echo "Press any key to close..."
read -n 1
