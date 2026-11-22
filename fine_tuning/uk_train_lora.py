import torch

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
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
    print("NLLB → UTTAR KARNATAKA LoRA FINE-TUNING")
    print("=" * 60)

    # --------------------------------------------------
    # GPU CHECK
    # --------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    device = torch.device("cuda")

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

    uk_config.UK_LORA_OUTPUT_DIR.mkdir(
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
    # LOAD BASE NLLB MODEL
    # --------------------------------------------------

    print("Loading base NLLB model...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        uk_config.MODEL_NAME,
        dtype=torch.float32,
    )

    model = model.to(device)

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
    # LoRA CONFIGURATION
    # --------------------------------------------------

    print("\nConfiguring LoRA...")

    lora_config = LoraConfig(

        task_type=TaskType.SEQ_2_SEQ_LM,

        r=uk_config.LORA_R,

        lora_alpha=uk_config.LORA_ALPHA,

        lora_dropout=uk_config.LORA_DROPOUT,

        target_modules=list(
            uk_config.LORA_TARGET_MODULES
        ),

        bias="none",
    )

    # --------------------------------------------------
    # ADD LoRA ADAPTERS
    # --------------------------------------------------

    print("Adding LoRA adapters...")

    model = get_peft_model(
        model,
        lora_config,
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
            uk_config.UK_LORA_OUTPUT_DIR
        ),

        num_train_epochs=(
            uk_config.LORA_NUM_TRAIN_EPOCHS
        ),

        per_device_train_batch_size=(
            uk_config.LORA_TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            uk_config.LORA_EVAL_BATCH_SIZE
        ),

        learning_rate=(
            uk_config.LORA_LEARNING_RATE
        ),

        weight_decay=(
            uk_config.WEIGHT_DECAY
        ),

        logging_steps=1,

        eval_strategy="epoch",

        # Save final adapter manually.
        save_strategy="no",

        fp16=False,

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
    print("STARTING LoRA FINE-TUNING")
    print("=" * 60 + "\n")

    trainer.train()

    # --------------------------------------------------
    # SAVE LoRA ADAPTER
    # --------------------------------------------------

    print("\nSaving LoRA adapter...")

    model.save_pretrained(
        str(uk_config.UK_LORA_OUTPUT_DIR)
    )

    tokenizer.save_pretrained(
        str(uk_config.UK_LORA_OUTPUT_DIR)
    )

    model.generation_config.save_pretrained(
        str(uk_config.UK_LORA_OUTPUT_DIR)
    )

    # --------------------------------------------------
    # COMPLETED
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("LoRA FINE-TUNING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"\nLoRA adapter saved to:\n"
        f"{uk_config.UK_LORA_OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()