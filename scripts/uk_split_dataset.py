import json
import random
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

# IMPORTANT:
# Split the CLEANED dataset, not the raw dataset.
INPUT_FILE = Path(
    "data/uk_cleaned/uk_cleaned.jsonl"
)

OUTPUT_DIR = Path("data/uk_dataset")

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
EVAL_FILE = OUTPUT_DIR / "eval.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TRAIN_RATIO = 0.80
EVAL_RATIO = 0.10
TEST_RATIO = 0.10


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    samples = []

    with open(
        INPUT_FILE,
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
# SAVE DATASET
# ============================================================

def save_dataset(file_path: Path, samples):

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as file:

        for sample in samples:

            file.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("UTTAR KARNATAKA DATASET SPLIT")
    print("=" * 60)

    print(f"\nInput dataset: {INPUT_FILE}")

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nCleaned dataset not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Load cleaned dataset
    # --------------------------------------------------------

    samples = load_dataset()

    total_samples = len(samples)

    print(f"Total cleaned samples: {total_samples}")

    if total_samples < 10:

        raise ValueError(
            "Dataset is too small to split properly."
        )

    # --------------------------------------------------------
    # Shuffle dataset reproducibly
    # --------------------------------------------------------

    random.seed(SEED)

    random.shuffle(samples)

    # --------------------------------------------------------
    # Calculate split sizes
    # --------------------------------------------------------

    train_size = int(
        total_samples * TRAIN_RATIO
    )

    eval_size = int(
        total_samples * EVAL_RATIO
    )

    # Remaining samples go to test
    test_size = (
        total_samples
        - train_size
        - eval_size
    )

    # --------------------------------------------------------
    # Split dataset
    # --------------------------------------------------------

    train_samples = samples[
        :train_size
    ]

    eval_samples = samples[
        train_size:
        train_size + eval_size
    ]

    test_samples = samples[
        train_size + eval_size:
    ]

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    assert (
        len(train_samples)
        + len(eval_samples)
        + len(test_samples)
        == total_samples
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    save_dataset(
        TRAIN_FILE,
        train_samples,
    )

    save_dataset(
        EVAL_FILE,
        eval_samples,
    )

    save_dataset(
        TEST_FILE,
        test_samples,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nDataset split completed successfully ✓")

    print(
        f"\nTrain samples: {len(train_samples)}"
    )

    print(
        f"Eval samples:  {len(eval_samples)}"
    )

    print(
        f"Test samples:  {len(test_samples)}"
    )

    print(
        f"\nTotal check: "
        f"{len(train_samples) + len(eval_samples) + len(test_samples)}"
    )

    print("\nFiles created:")

    print(TRAIN_FILE)
    print(EVAL_FILE)
    print(TEST_FILE)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()