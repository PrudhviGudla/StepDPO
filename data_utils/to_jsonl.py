from datasets import load_dataset
import json
import os

# Set your desired directory for storing the dataset
cache_dir = "./data"

# Download the dataset to the specified directory
ds = load_dataset("Birchlabs/openai-prm800k-stepwise-critic", cache_dir=cache_dir)

# print(ds)
# print(ds['train'][0])

output_dir = "../data/prm800k_stepwise_critic_split"
os.makedirs(output_dir, exist_ok=True)

# Function to save a split to jsonl
def save_jsonl(split, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in split:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Save train split
save_jsonl(ds["train"], os.path.join(output_dir, "train.jsonl"))

# Save test or validation split (if test not present)
split_name = "test" if "test" in ds else "validation"
save_jsonl(ds[split_name], os.path.join(output_dir, "test.jsonl"))

print("Done! Files saved to:", output_dir)