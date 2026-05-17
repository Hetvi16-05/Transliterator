#!/usr/bin/env bash
echo "══════════════════════════════════════"
echo "  Training: Model1-RuleBased"
echo "══════════════════════════════════════"
cd "/Users/HetviSheth/Transliterator"
python3 "/Users/HetviSheth/Transliterator/models/model1_rulebased.py" 2>&1 | tee "/Users/HetviSheth/Transliterator/logs/model1_rulebased.py.log"
echo ""
echo "✅ Model1-RuleBased — DONE  |  Log: /Users/HetviSheth/Transliterator/logs/model1_rulebased.py.log"
echo "Press any key to close..."
read -n 1
