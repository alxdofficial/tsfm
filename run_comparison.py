#!/usr/bin/env python3
"""
Simple wrapper to run the model comparison with proper error handling and output.
"""
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run
from training_scripts.imu_tool_pretraining import compare_models

if __name__ == '__main__':
    # Set up arguments
    sys.argv = [
        'compare_models.py',
        '--models',
        'original=training_output/semantic_alignment/original/best.pt',
        'shuffling_off=training_output/semantic_alignment/shufllingoff/best.pt',
        '--unseen_datasets', 'motionsense',
        '--output_dir', 'training_output/model_comparison_zeroshot'
    ]

    print("Running model comparison with the following arguments:")
    print(f"  Models: original, shuffling_off")
    print(f"  Unseen datasets: motionsense")
    print(f"  Output dir: training_output/model_comparison_zeroshot")
    print()

    try:
        compare_models.main()
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
