#!/bin/bash

# Set environment
export PATH="/home/Jennifer/miniconda3/bin:$PATH"
cd /home/Jennifer/projects/trader

# Run trading system
python enhanced_paper_trading.py --daily >> logs/cron.log 2>&1

# Send notification (optional)
if [ $? -eq 0 ]; then
    echo "Trading completed successfully at $(date)" >> logs/cron.log
else
    echo "Trading failed at $(date)" >> logs/cron.log
fi
