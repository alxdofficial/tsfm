import pandas as pd
import random
import itertools
import json

# --- Template Variations for Language Diversity ---

# Type 1: Simple questions about a single activity
Q_TYPE1_TEMPLATES = [
    "What is the person doing in this session?",
    "Can you describe the activity being performed?",
    "What is the primary task shown in this recording?",
    "Identify the main action taking place.",
    "What is happening in this clip?",
]

A_TYPE1_TEMPLATES = [
    "The person is {activity}.",
    "They are performing the task of {activity}.",
    "This is a recording of someone engaged in {activity}.",
    "The main activity is {activity}.",
    "The person is busy with {activity}.",
]

# Type 2: More complex questions spanning multiple activities
Q_TYPE2_ORDER_TEMPLATES = [
    "What is the correct order of the following kitchen chores: {shuffled_list}?",
    "Please sequence these tasks that the person performed: {shuffled_list}.",
    "The person did the following activities: {shuffled_list}. Can you put them in chronological order?",
]

A_TYPE2_ORDER_TEMPLATES = [
    "The correct order is: {correct_list}.",
    "They were performed in this sequence: {correct_list}.",
    "The chronological order of the tasks is: {correct_list}.",
]

Q_TYPE2_EXISTENCE_TEMPLATES = [
    "Did the person {activity} during this period?",
    "Was the task of {activity} performed in this set of activities?",
    "Is there evidence of the person doing {activity} in these sessions?",
]

Q_TYPE2_RELATIVE_ORDER_TEMPLATES = [
    "Was {act1} done before or after {act2}?",
    "Between {act1} and {act2}, which task came first?",
    "Can you confirm if {act1} preceded {act2}?",
]


def generate_qa_pairs(manifest_df: pd.DataFrame, all_activity_names: list, num_simple_target: int) -> tuple[list, list]:
    """
    Generates a list of question-answer pairs from the manifest data.
    """
    simple_qa_records = []
    complex_qa_records = []

    # --- Generate Type 1 Questions (Simple Activity Recognition) ---
    if not manifest_df.empty:
        while len(simple_qa_records) < num_simple_target:
            for _, row in manifest_df.iterrows():
                if len(simple_qa_records) >= num_simple_target:
                    break
                question = random.choice(Q_TYPE1_TEMPLATES)
                answer = random.choice(A_TYPE1_TEMPLATES).format(activity=row["activity_name"])

                simple_qa_records.append({
                    "question_type": "simple_activity_recognition",
                    "subject": row["subject"],
                    "split": row["split"],
                    "activity_indices": [row["activity_index"]],
                    "activity_names": [row["activity_name"]],
                    "question": question,
                    "answer": answer,
                })

    # --- Generate Type 2 Questions (Multi-Session Reasoning) ---
    grouped = manifest_df.groupby(["subject", "split"])

    for _, group in grouped:
        session_activities = group.sort_values("activity_index").to_dict("records")
        
        for k in range(2, min(len(session_activities), 4) + 1):
            for combo in itertools.combinations(session_activities, k):
                
                combo_indices = [act["activity_index"] for act in combo]
                combo_names = [act["activity_name"] for act in combo]

                # --- FIX: Ensure all activity names in the combination are unique ---
                if len(set(combo_names)) != len(combo_names):
                    continue

                # --- Ordering Question ---
                shuffled_names = random.sample(combo_names, len(combo_names))
                q_order = random.choice(Q_TYPE2_ORDER_TEMPLATES).format(shuffled_list=", ".join(shuffled_names))
                a_order = random.choice(A_TYPE2_ORDER_TEMPLATES).format(correct_list=", ".join(combo_names))
                complex_qa_records.append({
                    "question_type": "multi_activity_ordering",
                    "subject": combo[0]["subject"],
                    "split": combo[0]["split"],
                    "activity_indices": combo_indices,
                    "activity_names": combo_names,
                    "question": q_order,
                    "answer": a_order,
                })

                # --- Existence Questions (Positive and Negative) ---
                activity_to_find = random.choice(combo_names)
                q_exist_pos = random.choice(Q_TYPE2_EXISTENCE_TEMPLATES).format(activity=activity_to_find)
                complex_qa_records.append({
                    "question_type": "multi_activity_existence",
                    "subject": combo[0]["subject"],
                    "split": combo[0]["split"],
                    "activity_indices": combo_indices,
                    "activity_names": combo_names,
                    "question": q_exist_pos,
                    "answer": "Yes.",
                })

                other_activities = [name for name in all_activity_names if name not in combo_names]
                if other_activities:
                    activity_not_found = random.choice(other_activities)
                    q_exist_neg = random.choice(Q_TYPE2_EXISTENCE_TEMPLATES).format(activity=activity_not_found)
                    complex_qa_records.append({
                        "question_type": "multi_activity_existence",
                        "subject": combo[0]["subject"],
                        "split": combo[0]["split"],
                        "activity_indices": combo_indices,
                        "activity_names": combo_names,
                        "question": q_exist_neg,
                        "answer": "No.",
                    })

                # --- Relative Ordering Question ---
                if len(combo) >= 2:
                    act1, act2 = random.sample(combo, 2)
                    if act1["activity_index"] > act2["activity_index"]:
                        act1, act2 = act2, act1
                    
                    q_rel_order = random.choice(Q_TYPE2_RELATIVE_ORDER_TEMPLATES).format(act1=act1["activity_name"], act2=act2["activity_name"])
                    answer_rel_order = f'{act1["activity_name"]} was done before {act2["activity_name"]}.'
                    complex_qa_records.append({
                        "question_type": "multi_activity_relative_ordering",
                        "subject": combo[0]["subject"],
                        "split": combo[0]["split"],
                        "activity_indices": combo_indices,
                        "activity_names": combo_names,
                        "question": q_rel_order,
                        "answer": answer_rel_order,
                    })
    
    return simple_qa_records, complex_qa_records


def main():
    """
    Main function to run the dataset generation process.
    """
    manifest_path = "data/actionsenseqa_native/data/manifest.csv"
    output_path = "data/actionsenseqa_native/data/qa_pairs_templated.jsonl"
    target_total = 10000
    simple_ratio = 0.5 # 50% simple questions

    print(f"Reading manifest from: {manifest_path}")
    try:
        manifest_df = pd.read_csv(manifest_path)
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {manifest_path}")
        print("Please ensure the path is correct.")
        return

    all_activity_names = manifest_df["activity_name"].unique().tolist()

    print("Generating new templated QA pairs...")
    num_simple_target = int(target_total * simple_ratio)
    simple_qa, complex_qa = generate_qa_pairs(manifest_df, all_activity_names, num_simple_target)

    # --- Balanced Sampling ---
    num_complex_target = target_total - num_simple_target

    random.shuffle(simple_qa)
    random.shuffle(complex_qa)

    simple_sample = simple_qa[:min(len(simple_qa), num_simple_target)]
    complex_sample = complex_qa[:min(len(complex_qa), num_complex_target)]

    print(f"Sampling {len(simple_sample)} simple and {len(complex_sample)} complex pairs.")

    final_qa_data = simple_sample + complex_sample
    random.shuffle(final_qa_data)

    print(f"Saving {len(final_qa_data)} QA pairs to: {output_path}")
    with open(output_path, 'w') as f:
        for record in final_qa_data:
            f.write(json.dumps(record) + '\n')
    
    print("Done.")


if __name__ == "__main__":
    main()