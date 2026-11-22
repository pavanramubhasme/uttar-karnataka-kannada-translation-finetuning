import torch

from peft import PeftModel

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from fine_tuning.uk_config import uk_config


# ============================================================
# TRANSLATE
# ============================================================

def translate(text, model, tokenizer, device):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=uk_config.MAX_SOURCE_LENGTH,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    target_language_id = tokenizer.convert_tokens_to_ids(
        uk_config.TARGET_LANGUAGE
    )

    with torch.inference_mode():

        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=target_language_id,
            max_new_tokens=128,
            num_beams=5,
        )

    translation = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0]

    return translation


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NLLB → UTTAR KARNATAKA LoRA INFERENCE")
    print("=" * 60)

    # --------------------------------------------------
    # GPU CHECK
    # --------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    device = torch.device("cuda")

    print(
        f"\nGPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    # --------------------------------------------------
    # LOAD TOKENIZER
    # --------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        uk_config.UK_LORA_OUTPUT_DIR,
        src_lang=uk_config.SOURCE_LANGUAGE,
        tgt_lang=uk_config.TARGET_LANGUAGE,
    )

    # --------------------------------------------------
    # LOAD BASE MODEL
    # --------------------------------------------------

    print("Loading base NLLB model...")

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        uk_config.MODEL_NAME,
        dtype=torch.float32,
    )

    base_model = base_model.to(device)

    # --------------------------------------------------
    # LOAD LoRA ADAPTER
    # --------------------------------------------------

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        str(uk_config.UK_LORA_OUTPUT_DIR),
    )

    model = model.to(device)

    model.eval()

    # --------------------------------------------------
    # READY
    # --------------------------------------------------

    print("\nModel ready!")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:

        text = input("English: ").strip()

        if text.lower() in ["quit", "exit"]:

            print("Exiting...")
            break

        if not text:
            continue

        translation = translate(
            text,
            model,
            tokenizer,
            device,
        )

        print(
            f"\nUttar Karnataka: "
            f"{translation}\n"
        )


if __name__ == "__main__":
    main()