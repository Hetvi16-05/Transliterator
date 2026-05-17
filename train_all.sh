#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# train_all.sh  —  Launch all 8 models each in its own Terminal window
#
# Usage:  bash train_all.sh
# ──────────────────────────────────────────────────────────────────────────────

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$REPO/logs"
TMP_DIR="$REPO/.train_cmds"
mkdir -p "$LOG_DIR" "$REPO/saved_models" "$TMP_DIR"

PYTHON="/Applications/miniconda3/envs/ml/bin/python"

echo "🗂  Repo  : $REPO"
echo "🐍  Python: $($PYTHON --version 2>&1)"
echo "📁  Logs  : $LOG_DIR"
echo ""

declare -a NAMES=("Model1-RuleBased" "Model2-RNN" "Model3-LSTM" "Model4-BiLSTM" "Model5-GRU" "Model6-Attention" "Model7-Transformer" "Model8-CNN-LSTM")
declare -a SCRIPTS=("models/model1_rulebased.py" "models/model2_rnn.py" "models/model3_lstm.py" "models/model4_bilstm.py" "models/model5_gru.py" "models/model6_attention.py" "models/model7_transformer.py" "models/model8_cnn_lstm.py")

echo "🚀  Launching ${#NAMES[@]} training jobs in separate Terminal windows..."
echo ""

for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    script="${SCRIPTS[$i]}"
    logfile="$LOG_DIR/${script//\//_}.log"
    logfile="${logfile//models_/}"      # tidy up the log name
    cmdfile="$TMP_DIR/${name}.command"

    # Write a self-contained .command script for this model
    cat > "$cmdfile" <<CMDEOF
#!/usr/bin/env bash
echo "══════════════════════════════════════"
echo "  Training: $name"
echo "══════════════════════════════════════"
cd "$REPO"
$PYTHON "$REPO/$script" 2>&1 | tee "$logfile"
echo ""
echo "✅ $name — DONE  |  Log: $logfile"
echo "Press any key to close..."
read -n 1
CMDEOF
    chmod +x "$cmdfile"

    # Open in a new Terminal window
    open -a Terminal "$cmdfile"
    echo "✅  Launched: $name"
    sleep 0.5
done

echo ""
echo "──────────────────────────────────────────"
echo "All ${#NAMES[@]} training jobs launched!"
echo "Monitor logs in:  $LOG_DIR/"
echo "──────────────────────────────────────────"
