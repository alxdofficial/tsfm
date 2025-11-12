"""
Test the plotting utilities.

Run this to verify the plotting system works before training.
"""

from pathlib import Path
import shutil
from plot_utils import TrainingPlotter


def test_plotting():
    """Test plot generation with dummy data."""
    print("Testing TrainingPlotter...")

    # Create temporary output directory
    test_dir = Path("test_plots")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    # Create plotter
    plotter = TrainingPlotter(output_dir=test_dir)

    # Simulate training for 20 epochs with 4 datasets
    datasets = ['uci_har', 'mhealth', 'pamap2', 'wisdm']

    for epoch in range(20):
        # Simulate decreasing losses
        train_loss = 2.0 - epoch * 0.08 + 0.1 * (epoch % 3)
        val_loss = 2.1 - epoch * 0.07 + 0.15 * (epoch % 3)
        mae_loss_train = train_loss * 0.6
        mae_loss_val = val_loss * 0.6
        contrast_loss_train = train_loss * 0.4
        contrast_loss_val = val_loss * 0.4
        lr = 1e-4 * (1 - epoch / 20)  # Decaying learning rate

        # Log epoch metrics
        plotter.add_scalar('epoch/train_loss', train_loss, epoch)
        plotter.add_scalar('epoch/val_loss', val_loss, epoch)
        plotter.add_scalar('epoch/train_mae_loss', mae_loss_train, epoch)
        plotter.add_scalar('epoch/val_mae_loss', mae_loss_val, epoch)
        plotter.add_scalar('epoch/train_contrastive_loss', contrast_loss_train, epoch)
        plotter.add_scalar('epoch/val_contrastive_loss', contrast_loss_val, epoch)
        plotter.add_scalar('epoch/lr', lr, epoch)

        # Log per-dataset metrics (with some variation)
        for i, dataset in enumerate(datasets):
            dataset_train_loss = train_loss * (1.0 + 0.1 * i)
            dataset_val_loss = val_loss * (1.0 + 0.1 * i)

            plotter.add_scalar(f'train_per_dataset/{dataset}/loss', dataset_train_loss, epoch)
            plotter.add_scalar(f'train_per_dataset/{dataset}/mae_loss', dataset_train_loss * 0.6, epoch)
            plotter.add_scalar(f'train_per_dataset/{dataset}/contrastive_loss', dataset_train_loss * 0.4, epoch)

            plotter.add_scalar(f'val_per_dataset/{dataset}/loss', dataset_val_loss, epoch)
            plotter.add_scalar(f'val_per_dataset/{dataset}/mae_loss', dataset_val_loss * 0.6, epoch)
            plotter.add_scalar(f'val_per_dataset/{dataset}/contrastive_loss', dataset_val_loss * 0.4, epoch)

    # Generate all plots
    print("\nGenerating plots...")
    plotter.close()

    # Check that all expected files exist
    expected_files = [
        'overall_loss.png',
        'loss_components.png',
        'per_dataset_losses.png',
        'learning_rate.png',
        'metrics.json',
    ] + [f'dataset_{ds}_detail.png' for ds in datasets]

    print("\nChecking generated files:")
    all_exist = True
    for fname in expected_files:
        fpath = test_dir / fname
        exists = fpath.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {fname}")
        if not exists:
            all_exist = False

    if all_exist:
        print("\n✓ All plots generated successfully!")
        print(f"\nCheck the plots in: {test_dir.absolute()}")
        print("\nTo clean up test files, run:")
        print(f"  rm -rf {test_dir}")
    else:
        print("\n✗ Some plots failed to generate")
        return False

    return True


if __name__ == "__main__":
    success = test_plotting()
    exit(0 if success else 1)
