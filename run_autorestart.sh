#!/bin/bash

# Auto-restart training script
# This script will continuously restart the training command when it stops

LOG_FILE="training_restart.log"
MAX_RESTARTS=100
RESTART_COUNT=0

echo "Starting auto-restart training script at $(date)" | tee -a $LOG_FILE
echo "Training command: python scripts/train.py -E exp/exp_043_abl_dvc_only --gpus 0" | tee -a $LOG_FILE

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    echo "=== Restart attempt $((RESTART_COUNT + 1)) at $(date) ===" | tee -a $LOG_FILE
    
    # Run the training command
    python scripts/train.py -E exp/exp_043_abl_dvc_only --gpus 0
    
    # Capture the exit code
    EXIT_CODE=$?
    
    echo "Training stopped with exit code: $EXIT_CODE at $(date)" | tee -a $LOG_FILE
    
    # Check if it was a clean exit (0) or an error
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully. Exiting." | tee -a $LOG_FILE
        break
    else
        echo "Training failed or was interrupted. Restarting in 10 seconds..." | tee -a $LOG_FILE
        sleep 10
        RESTART_COUNT=$((RESTART_COUNT + 1))
    fi
done

if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
    echo "Maximum restart attempts ($MAX_RESTARTS) reached. Stopping." | tee -a $LOG_FILE
fi

echo "Auto-restart script finished at $(date)" | tee -a $LOG_FILE