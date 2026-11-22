import torch

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


def main():

    print("=" * 60)
    print("NLLB → UTTAR KARNATAKA FULL FINE-TUNING")
    print("=" * 60)

    # --------------------------------------------------
    # GPU CHECK
    # --------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    device_name = torch.cuda.get_device_name(0)

    print(f"\nGPU: {device_name}")

    # --------------------------------------------------
    # SEED
    # --------------------------------------------------

    set_seed(uk_config.SEED)

    # --------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------

    uk_config.UK_FULL_OUTPUT_DIR.mkdir(
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
    # LOAD MODEL
    # --------------------------------------------------

    print("Loading NLLB model...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        uk_config.MODEL_NAME,
        torch_dtype=torch.float32,
    )

    # Move model to GPU
    model = model.to("cuda")

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
    # LOAD DATASETS
    # --------------------------------------------------

    print("\nLoading datasets...")

    train_dataset, eval_dataset = get_uk_datasets()

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

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
            text_target=examples["uttar_karnataka"],
            max_length=uk_config.MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]

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
            uk_config.UK_FULL_OUTPUT_DIR
        ),

        num_train_epochs=(
            uk_config.FULL_NUM_TRAIN_EPOCHS
        ),

        per_device_train_batch_size=(
            uk_config.FULL_TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            uk_config.FULL_EVAL_BATCH_SIZE
        ),

        learning_rate=(
            uk_config.FULL_LEARNING_RATE
        ),

        weight_decay=uk_config.WEIGHT_DECAY,

        logging_steps=1,

        eval_strategy="epoch",

        # Save only the final model manually.
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
    print("STARTING FULL FINE-TUNING")
    print("=" * 60 + "\n")

    trainer.train()

    # --------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------

    print("\nSaving full fine-tuned model...")

    model.config.forced_bos_token_id = None

    model.generation_config.forced_bos_token_id = (
        tokenizer.convert_tokens_to_ids(
            uk_config.TARGET_LANGUAGE
        )
    )

    trainer.save_model(
        str(uk_config.UK_FULL_OUTPUT_DIR)
    )

    tokenizer.save_pretrained(
        str(uk_config.UK_FULL_OUTPUT_DIR)
    )

    model.generation_config.save_pretrained(
        str(uk_config.UK_FULL_OUTPUT_DIR)
    )

    print("\n" + "=" * 60)
    print("FULL FINE-TUNING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"\nModel saved to:\n"
        f"{uk_config.UK_FULL_OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()