#!/usr/bin/env bash
echo "══════════════════════════════════════"
echo "  Training: Model6-Attention"
echo "══════════════════════════════════════"
cd "/Users/HetviSheth/Transliterator"
python3 "/Users/HetviSheth/Transliterator/models/model6_attention.py" 2>&1 | tee "/Users/HetviSheth/Transliterator/logs/model6_attention.py.log"
echo ""
echo "✅ Model6-Attention — DONE  |  Log: /Users/HetviSheth/Transliterator/logs/model6_attention.py.log"
echo "Press any key to close..."
read -n 1
