import torch

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

    with torch.no_grad():

        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=target_language_id,
            max_new_tokens=uk_config.MAX_TARGET_LENGTH,
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
    print("NLLB → UTTAR KARNATAKA FULL FINE-TUNING INFERENCE")
    print("=" * 60)

    # --------------------------------------------------
    # GPU CHECK
    # --------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    device = torch.device("cuda")

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------
    # LOAD TOKENIZER
    # --------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        uk_config.UK_FULL_OUTPUT_DIR,
        src_lang=uk_config.SOURCE_LANGUAGE,
        tgt_lang=uk_config.TARGET_LANGUAGE,
    )

    # --------------------------------------------------
    # LOAD FULL FINE-TUNED MODEL
    # --------------------------------------------------

    print("Loading full fine-tuned model...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        uk_config.UK_FULL_OUTPUT_DIR,
        torch_dtype=torch.float32,
    )

    model = model.to(device)
    model.eval()

    # --------------------------------------------------
    # READY
    # --------------------------------------------------

    print("\nModel ready!")
    print("Type 'quit' or 'exit' to stop.\n")

    # --------------------------------------------------
    # INTERACTIVE INFERENCE
    # --------------------------------------------------

    while True:

        text = input("English: ").strip()

        if text.lower() in ["quit", "exit"]:
            print("Exiting...")
            break

        if not text:
            continue

        translation = translate(
            text=text,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        print(f"\nUttar Karnataka: {translation}\n")


if __name__ == "__main__":
    main()