import torch

from peft import PeftModel

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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
    )[0].strip()

    return translation


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NLLB → UTTAR KARNATAKA QLoRA INFERENCE")
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
    # LOAD TOKENIZER
    # --------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        uk_config.UK_QLORA_OUTPUT_DIR,
        src_lang=uk_config.SOURCE_LANGUAGE,
        tgt_lang=uk_config.TARGET_LANGUAGE,
    )

    # --------------------------------------------------
    # 4-BIT QUANTIZATION CONFIG
    # --------------------------------------------------

    print("Configuring 4-bit quantization...")

    quantization_config = BitsAndBytesConfig(

        load_in_4bit=True,

        bnb_4bit_quant_type=(
            uk_config.QLORA_QUANT_TYPE
        ),

        bnb_4bit_compute_dtype=torch.float16,

        bnb_4bit_use_double_quant=(
            uk_config.QLORA_USE_DOUBLE_QUANT
        ),
    )

    # --------------------------------------------------
    # LOAD BASE MODEL IN 4-BIT
    # --------------------------------------------------

    print("Loading base NLLB model in 4-bit...")

    base_model = AutoModelForSeq2SeqLM.from_pretrained(

        uk_config.MODEL_NAME,

        quantization_config=quantization_config,

        device_map="auto",
    )

    # --------------------------------------------------
    # LOAD QLoRA ADAPTER
    # --------------------------------------------------

    print("Loading QLoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        str(uk_config.UK_QLORA_OUTPUT_DIR),
    )

    model.eval()

    # --------------------------------------------------
    # READY
    # --------------------------------------------------

    print("\nModel ready!")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:

        text = input("English: ").strip()

        if text.lower() in [
            "quit",
            "exit",
        ]:

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