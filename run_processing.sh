#!/bin/bash
cd /home/alex/code/tsfm
echo "Starting MotionSense processing..."
python3 datascripts/process_motionsense.py
echo "Done with exit code: $?"
