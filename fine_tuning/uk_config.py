from dataclasses import dataclass
from pathlib import Path


@dataclass
class UKConfig:

    # ============================================================
    # MODEL
    # ============================================================

    MODEL_NAME: str = "facebook/nllb-200-distilled-600M"

    SOURCE_LANGUAGE: str = "eng_Latn"
    TARGET_LANGUAGE: str = "kan_Knda"


    # ============================================================
    # PROJECT PATHS
    # ============================================================

    PROJECT_ROOT: Path = (
        Path(__file__).resolve().parent.parent
    )

    DATA_DIR: Path = PROJECT_ROOT / "data"

    UK_DATA_DIR: Path = (
        DATA_DIR / "uk_dataset"
    )


    # ============================================================
    # DATASET FILES
    # ============================================================

    UK_TRAIN_FILE: Path = (
        UK_DATA_DIR / "train.jsonl"
    )

    UK_EVAL_FILE: Path = (
        UK_DATA_DIR / "eval.jsonl"
    )

    UK_TEST_FILE: Path = (
        UK_DATA_DIR / "test.jsonl"
    )


    # ============================================================
    # MODEL OUTPUTS
    # ============================================================

    UK_FULL_OUTPUT_DIR: Path = (
        PROJECT_ROOT
        / "outputs"
        / "models"
        / "uk_nllb_full"
    )

    UK_LORA_OUTPUT_DIR: Path = (
        PROJECT_ROOT
        / "outputs"
        / "models"
        / "uk_nllb_lora"
    )

    UK_QLORA_OUTPUT_DIR: Path = (
        PROJECT_ROOT
        / "outputs"
        / "models"
        / "uk_nllb_qlora"
    )


    # ============================================================
    # DATA
    # ============================================================

    MAX_SOURCE_LENGTH: int = 128

    MAX_TARGET_LENGTH: int = 128


    # ============================================================
    # FULL FINE-TUNING
    # ============================================================

    FULL_NUM_TRAIN_EPOCHS: int = 10

    FULL_TRAIN_BATCH_SIZE: int = 2

    FULL_EVAL_BATCH_SIZE: int = 2

    FULL_LEARNING_RATE: float = 5e-5


    # ============================================================
    # LoRA FINE-TUNING
    # ============================================================

    LORA_NUM_TRAIN_EPOCHS: int = 10

    LORA_TRAIN_BATCH_SIZE: int = 2

    LORA_EVAL_BATCH_SIZE: int = 2

    LORA_LEARNING_RATE: float = 1e-4

    LORA_R: int = 16

    LORA_ALPHA: int = 32

    LORA_DROPOUT: float = 0.05

    LORA_TARGET_MODULES: tuple = (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    )


    # ============================================================
    # QLoRA FINE-TUNING
    # ============================================================

    QLORA_NUM_TRAIN_EPOCHS: int = 10

    QLORA_TRAIN_BATCH_SIZE: int = 2

    QLORA_EVAL_BATCH_SIZE: int = 2

    QLORA_LEARNING_RATE: float = 1e-4

    QLORA_R: int = 16

    QLORA_ALPHA: int = 32

    QLORA_DROPOUT: float = 0.05

    QLORA_TARGET_MODULES: tuple = (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    )


    # ============================================================
    # QLoRA QUANTIZATION
    # ============================================================

    QLORA_LOAD_IN_4BIT: bool = True

    QLORA_QUANT_TYPE: str = "nf4"

    QLORA_USE_DOUBLE_QUANT: bool = True


    # ============================================================
    # GENERAL TRAINING
    # ============================================================

    WEIGHT_DECAY: float = 0.01

    SEED: int = 42


uk_config = UKConfig()