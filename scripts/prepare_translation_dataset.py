from pathlib import Path
from datasets import load_dataset

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

TRAIN_FILE = DATA_DIR / "train.jsonl"
EVAL_FILE = DATA_DIR / "eval.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"

SEED = 42

# Start with a manageable dataset size
MAX_SAMPLES = 12000
TRAIN_SIZE = 10000
EVAL_SIZE = 1000
TEST_SIZE = 1000

def main():
    print("=" * 60)
    print("ENGLISH → KANNADA DATASET PREPARATION")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    print("\nLoading English-Kannada dataset...")

    dataset = load_dataset(
        "ai4bharat/samanantar",
        "kn",
        split="train",
    )

    print(f"Total available samples: {len(dataset)}")
    # --------------------------------------------------------
    # Select samples
    # --------------------------------------------------------
    print(f"\nSelecting {MAX_SAMPLES} samples...")
    dataset = dataset.shuffle(seed=SEED)
    dataset = dataset.select(
        range(min(MAX_SAMPLES, len(dataset)))
    )

    # --------------------------------------------------------
    # Format dataset
    # --------------------------------------------------------

    def format_example(example):
        english = example["src"].strip()
        kannada = example["tgt"].strip()

        return {
            "english": english,
            "kannada": kannada,
        }

    print("\nFormatting translation pairs...")
    dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
    )

    # --------------------------------------------------------
    # Remove empty samples
    # --------------------------------------------------------

    print("Removing invalid samples...")
    dataset = dataset.filter(
        lambda x:
        len(x["english"]) > 0
        and len(x["kannada"]) > 0
    )
    print(f"Valid samples: {len(dataset)}")

    # --------------------------------------------------------
    # Create splits
    # --------------------------------------------------------
    print("\nCreating train/evaluation/test splits...")
    if len(dataset) < MAX_SAMPLES:
        raise ValueError(
            f"Not enough valid samples. "
            f"Expected {MAX_SAMPLES}, got {len(dataset)}"
        )

    train_dataset = dataset.select(
        range(0, TRAIN_SIZE)
    )

    eval_dataset = dataset.select(
        range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)
    )

    test_dataset = dataset.select(
        range(
            TRAIN_SIZE + EVAL_SIZE,
            TRAIN_SIZE + EVAL_SIZE + TEST_SIZE,
        )
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------
    print("\nSaving datasets...")
    train_dataset.to_json(TRAIN_FILE)
    eval_dataset.to_json(EVAL_FILE)
    test_dataset.to_json(TEST_FILE)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 60)

    print(f"\nTraining samples:   {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")

    print("\nFiles created:")

    print(f"Train: {TRAIN_FILE}")
    print(f"Eval:  {EVAL_FILE}")
    print(f"Test:  {TEST_FILE}")

    print("\nExample:")

    example = train_dataset[0]

    print(f"\nEnglish: {example['english']}")
    print(f"Kannada: {example['kannada']}")


if __name__ == "__main__":
    main()