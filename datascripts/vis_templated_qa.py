
import pandas as pd
import json
import random
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_sensor_data(csv_path: str, base_dir: str) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """
    Loads a single sensor CSV file.
    """
    full_path = os.path.join(base_dir, csv_path)
    if not os.path.exists(full_path):
        print(f"[WARN] Sensor CSV missing: {full_path}")
        return None
    
    df = pd.read_csv(full_path)
    if "time_s" not in df.columns:
        print(f"[WARN] Sensor CSV missing 'time_s' column: {full_path}")
        return None

    timestamps = df["time_s"].to_numpy(dtype=np.float64)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "time_s"]
    values = df[numeric_cols].to_numpy(dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    
    return timestamps, values, numeric_cols

def plot_single_activity_sample(sample: dict, sensor_data: dict, out_dir: str, sample_num: int):
    """
    Visualizes a single-activity (Type 1) QA pair.
    """
    ts, vals, cols = sensor_data[sample["activity_indices"][0]]

    plt.figure(figsize=(12, 5))
    for d in range(min(10, vals.shape[1])):
        label = cols[d] if cols and d < len(cols) else f"ch{d}"
        plt.plot(ts, vals[:, d], label=label)

    plt.title(f"Q: {sample['question']}\nA: {sample['answer']}", wrap=True)
    plt.xlabel("time_s")
    plt.ylabel("Sensor Value")
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()

    save_path = os.path.join(out_dir, f"sample_{sample_num}_single_activity.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[PLOT] Saved single-activity plot to {save_path}")

def plot_multi_activity_sample(sample: dict, sensor_data: dict, out_dir: str, sample_num: int):
    """
    Visualizes a multi-activity (Type 2) QA pair with subplots.
    """
    num_activities = len(sample["activity_indices"])
    fig, axes = plt.subplots(num_activities, 1, figsize=(12, 3 * num_activities), sharex=True)
    if num_activities == 1:
        axes = [axes] # Make it iterable

    fig.suptitle(f"Q: {sample['question']}\nA: {sample['answer']}", fontsize=14, wrap=True)

    for i, activity_index in enumerate(sample["activity_indices"]):
        ax = axes[i]
        ts, vals, cols = sensor_data[activity_index]
        activity_name = sample["activity_names"][i]

        for d in range(min(10, vals.shape[1])):
            label = cols[d] if cols and d < len(cols) else f"ch{d}"
            ax.plot(ts, vals[:, d], label=label)
        
        ax.set_title(f"Session {activity_index}: {activity_name}")
        ax.set_ylabel("Sensor Value")
        ax.legend(loc="upper right", ncol=2)

    axes[-1].set_xlabel("time_s")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(out_dir, f"sample_{sample_num}_multi_activity.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[PLOT] Saved multi-activity plot to {save_path}")


def main(args):
    """
    Main function to run the visualization.
    """
    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)

    # Load the open-ended QA dataset
    qa_pairs = []
    try:
        with open(args.qa_path, 'r') as f:
            for line in f:
                qa_pairs.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: QA file not found at {args.qa_path}")
        return

    # Filter by question type if specified
    if args.question_type:
        qa_pairs = [p for p in qa_pairs if p["question_type"] == args.question_type]
        if not qa_pairs:
            print(f"No samples found for question type: {args.question_type}")
            return

    # Load manifest to find sensor data paths
    try:
        manifest_df = pd.read_csv(args.manifest_path).set_index('activity_index')
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {args.manifest_path}")
        return
    manifest_map = manifest_df.to_dict('index')

    # Select random samples to visualize
    num_samples = min(args.num_samples, len(qa_pairs))
    random.seed(args.seed)
    selected_samples = random.sample(qa_pairs, num_samples)

    print(f"--- Visualizing {num_samples} random samples ---")

    for i, sample in enumerate(selected_samples):
        # Load all required sensor data for this sample
        sensor_data_cache = {}
        all_data_found = True
        for activity_index in sample["activity_indices"]:
            if activity_index not in manifest_map:
                print(f"[WARN] Activity index {activity_index} not found in manifest.")
                all_data_found = False
                break
            
            csv_path = manifest_map[activity_index]['csv_path']
            sensor_content = load_sensor_data(csv_path, args.base_dir)
            if sensor_content is None:
                all_data_found = False
                break
            sensor_data_cache[activity_index] = sensor_content
        
        if not all_data_found:
            print(f"Skipping sample {i} due to missing data.")
            continue

        # Plot based on question type
        if sample["question_type"] == "simple_activity_recognition":
            plot_single_activity_sample(sample, sensor_data_cache, args.outdir, i)
        else:
            plot_multi_activity_sample(sample, sensor_data_cache, args.outdir, i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize samples from the open-ended ActionSense QA dataset.")
    parser.add_argument("--qa_path", type=str, default="data/actionsenseqa/data/qa_pairs_templated.jsonl", help="Path to the generated .jsonl QA file.")
    parser.add_argument("--manifest_path", type=str, default="data/actionsenseqa/data/manifest.csv", help="Path to the manifest CSV file.")
    parser.add_argument("--base_dir", type=str, default="data/actionsenseqa/data", help="Base directory where sensor CSVs are stored.")
    parser.add_argument("--outdir", type=str, default="debug/templated_qa_visuals", help="Directory to save the output plots.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to visualize.")
    parser.add_argument("--question_type", type=str, default=None, help="Filter for a specific question type (e.g., multi_activity_ordering).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")

    args = parser.parse_args()
    main(args)
