from pathlib import Path
import json

from datasets import Dataset

from fine_tuning.uk_config import uk_config


# ============================================================
# LOAD JSONL FILE
# ============================================================

def load_jsonl(file_path: Path):

    samples = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            samples.append(
                json.loads(line)
            )

    return samples


# ============================================================
# LOAD UK DATASETS
# ============================================================

def get_uk_datasets():

    train_samples = load_jsonl(
        uk_config.UK_TRAIN_FILE
    )

    eval_samples = load_jsonl(
        uk_config.UK_EVAL_FILE
    )

    train_dataset = Dataset.from_list(
        train_samples
    )

    eval_dataset = Dataset.from_list(
        eval_samples
    )

    return train_dataset, eval_dataset


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    train_dataset, eval_dataset = get_uk_datasets()

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Eval samples: {len(eval_dataset)}"
    )

    print("\nSample:")

    if len(train_dataset) > 0:
        print(train_dataset[0])