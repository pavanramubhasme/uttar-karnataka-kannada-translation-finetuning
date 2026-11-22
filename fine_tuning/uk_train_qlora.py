import torch

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from fine_tuning.uk_config import uk_config
from fine_tuning.uk_dataset import get_uk_datasets


# ============================================================
# PRINT TRAINABLE PARAMETERS
# ============================================================

def print_trainable_parameters(model):

    trainable_params = 0
    total_params = 0

    for parameter in model.parameters():

        total_params += parameter.numel()

        if parameter.requires_grad:
            trainable_params += parameter.numel()

    trainable_percentage = (
        100 * trainable_params / total_params
    )

    print("\n" + "=" * 60)
    print("PARAMETER SUMMARY")
    print("=" * 60)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(
        f"Trainable percentage: "
        f"{trainable_percentage:.4f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NLLB → UTTAR KARNATAKA QLoRA FINE-TUNING")
    print("=" * 60)

    # --------------------------------------------------
    # GPU CHECK
    # --------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    print(
        f"\nGPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    # --------------------------------------------------
    # SEED
    # --------------------------------------------------

    set_seed(uk_config.SEED)

    # --------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------

    uk_config.UK_QLORA_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # LOAD TOKENIZER
    # --------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        uk_config.MODEL_NAME,
        src_lang=uk_config.SOURCE_LANGUAGE,
        tgt_lang=uk_config.TARGET_LANGUAGE,
    )

    # --------------------------------------------------
    # 4-BIT QUANTIZATION CONFIGURATION
    # --------------------------------------------------

    print("\nConfiguring 4-bit quantization...")

    quantization_config = BitsAndBytesConfig(

        load_in_4bit=True,

        bnb_4bit_quant_type=(
            uk_config.QLORA_QUANT_TYPE
        ),

        bnb_4bit_compute_dtype=(
            torch.float16
        ),

        bnb_4bit_use_double_quant=(
            uk_config.QLORA_USE_DOUBLE_QUANT
        ),
    )

    # --------------------------------------------------
    # LOAD BASE MODEL IN 4-BIT
    # --------------------------------------------------

    print("Loading NLLB model in 4-bit...")

    model = AutoModelForSeq2SeqLM.from_pretrained(

        uk_config.MODEL_NAME,

        quantization_config=quantization_config,

        device_map="auto",
    )

    # --------------------------------------------------
    # NLLB GENERATION CONFIGURATION
    # --------------------------------------------------

    model.config.forced_bos_token_id = None

    model.generation_config.forced_bos_token_id = (
        tokenizer.convert_tokens_to_ids(
            uk_config.TARGET_LANGUAGE
        )
    )

    # --------------------------------------------------
    # PREPARE MODEL FOR QLoRA TRAINING
    # --------------------------------------------------

    print(
        "\nPreparing model for k-bit training..."
    )

    model = prepare_model_for_kbit_training(
        model
    )

    # --------------------------------------------------
    # QLoRA CONFIGURATION
    # --------------------------------------------------

    print("Configuring QLoRA adapters...")

    qlora_config = LoraConfig(

        task_type=TaskType.SEQ_2_SEQ_LM,

        r=uk_config.QLORA_R,

        lora_alpha=uk_config.QLORA_ALPHA,

        lora_dropout=uk_config.QLORA_DROPOUT,

        target_modules=list(
            uk_config.QLORA_TARGET_MODULES
        ),

        bias="none",
    )

    # --------------------------------------------------
    # ADD QLoRA ADAPTERS
    # --------------------------------------------------

    print("Adding QLoRA adapters...")

    model = get_peft_model(
        model,
        qlora_config,
    )

    # --------------------------------------------------
    # PRINT TRAINABLE PARAMETERS
    # --------------------------------------------------

    print_trainable_parameters(model)

    # --------------------------------------------------
    # LOAD DATASETS
    # --------------------------------------------------

    print("\nLoading datasets...")

    train_dataset, eval_dataset = (
        get_uk_datasets()
    )

    print(
        f"Train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Eval samples: "
        f"{len(eval_dataset)}"
    )

    # --------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------

    print("\nTokenizing datasets...")

    def preprocess_function(examples):

        model_inputs = tokenizer(
            examples["english"],
            max_length=uk_config.MAX_SOURCE_LENGTH,
            truncation=True,
            padding=False,
        )

        labels = tokenizer(
            text_target=examples[
                "uttar_karnataka"
            ],
            max_length=uk_config.MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = (
            labels["input_ids"]
        )

        return model_inputs

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    # --------------------------------------------------
    # DATA COLLATOR
    # --------------------------------------------------

    data_collator = DataCollatorForSeq2Seq(

        tokenizer=tokenizer,

        model=model,

        label_pad_token_id=-100,
    )

    # --------------------------------------------------
    # TRAINING ARGUMENTS
    # --------------------------------------------------

    training_args = TrainingArguments(

        output_dir=str(
            uk_config.UK_QLORA_OUTPUT_DIR
        ),

        num_train_epochs=(
            uk_config.QLORA_NUM_TRAIN_EPOCHS
        ),

        per_device_train_batch_size=(
            uk_config.QLORA_TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            uk_config.QLORA_EVAL_BATCH_SIZE
        ),

        learning_rate=(
            uk_config.QLORA_LEARNING_RATE
        ),

        weight_decay=(
            uk_config.WEIGHT_DECAY
        ),

        logging_steps=1,

        eval_strategy="epoch",

        # Save final adapter manually
        save_strategy="no",

        fp16=True,

        bf16=False,

        report_to="none",

        seed=uk_config.SEED,

        dataloader_pin_memory=True,

        remove_unused_columns=True,
    )

    # --------------------------------------------------
    # TRAINER
    # --------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        data_collator=data_collator,

        processing_class=tokenizer,
    )

    # --------------------------------------------------
    # START TRAINING
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STARTING QLoRA FINE-TUNING")
    print("=" * 60 + "\n")

    trainer.train()

    # --------------------------------------------------
    # SAVE QLoRA ADAPTER
    # --------------------------------------------------

    print("\nSaving QLoRA adapter...")

    model.save_pretrained(
        str(uk_config.UK_QLORA_OUTPUT_DIR)
    )

    tokenizer.save_pretrained(
        str(uk_config.UK_QLORA_OUTPUT_DIR)
    )

    model.generation_config.save_pretrained(
        str(uk_config.UK_QLORA_OUTPUT_DIR)
    )

    # --------------------------------------------------
    # COMPLETED
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("QLoRA FINE-TUNING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"\nQLoRA adapter saved to:\n"
        f"{uk_config.UK_QLORA_OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()