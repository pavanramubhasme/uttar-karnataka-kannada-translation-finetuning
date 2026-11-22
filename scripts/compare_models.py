import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from fine_tuning.uk_config import uk_config


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "comparison"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "full_vs_lora_vs_qlora.json"
)


# ============================================================
# TEST SENTENCES
# ============================================================

# Try to use realistic sentences for qualitative comparison.
TEST_SENTENCES = [

    "You also bring your mobile with you.",

    "Did you sleep well? Are you feeling better now?",

    "I am very hungry now.",

    "Bring my phone here.",

    "Don't wait for me.",

    "I will call you later.",

    "What are you doing now?",

    "Are you free now?",

    "I will come tomorrow morning.",

    "Where are you going after work?",

    "Bring some water when you come.",

    "I am feeling very tired today.",

]


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(
        uk_config.UK_FULL_OUTPUT_DIR
    )

    tokenizer.src_lang = (
        uk_config.SOURCE_LANGUAGE
    )

    return tokenizer


# ============================================================
# TRANSLATE
# ============================================================

def translate(
    model,
    tokenizer,
    english_text,
):

    inputs = tokenizer(
        english_text,
        return_tensors="pt",
        max_length=uk_config.MAX_SOURCE_LENGTH,
        truncation=True,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        generated_tokens = model.generate(

            **inputs,

            forced_bos_token_id=(
                tokenizer.convert_tokens_to_ids(
                    uk_config.TARGET_LANGUAGE
                )
            ),

            max_new_tokens=128,

            num_beams=4,

            do_sample=False,
        )

    translation = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0]

    return translation.strip()


# ============================================================
# LOAD FULL FINE-TUNED MODEL
# ============================================================

def load_full_model():

    print("\nLoading FULL fine-tuned model...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        uk_config.UK_FULL_OUTPUT_DIR,
        torch_dtype=torch.float16,
    )

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# LOAD LORA MODEL
# ============================================================

def load_lora_model():

    print("\nLoading LoRA model...")

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        uk_config.MODEL_NAME,
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(
        base_model,
        uk_config.UK_LORA_OUTPUT_DIR,
    )

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# LOAD QLORA MODEL
# ============================================================

def load_qlora_model():

    print("\nLoading QLoRA model...")

    quantization_config = BitsAndBytesConfig(

        load_in_4bit=True,

        bnb_4bit_quant_type="nf4",

        bnb_4bit_compute_dtype=torch.float16,

        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForSeq2SeqLM.from_pretrained(

        uk_config.MODEL_NAME,

        quantization_config=quantization_config,

        device_map="auto",
    )

    model = PeftModel.from_pretrained(

        base_model,

        uk_config.UK_QLORA_OUTPUT_DIR,
    )

    model.eval()

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "FULL vs LoRA vs QLoRA "
        "QUALITATIVE COMPARISON"
    )

    print("=" * 70)

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = load_tokenizer()


    # --------------------------------------------------------
    # FULL MODEL
    # --------------------------------------------------------

    full_model = load_full_model()


    # --------------------------------------------------------
    # LORA MODEL
    # --------------------------------------------------------

    lora_model = load_lora_model()


    # --------------------------------------------------------
    # QLORA MODEL
    # --------------------------------------------------------

    qlora_model = load_qlora_model()


    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    results = []

    for index, english_text in enumerate(
        TEST_SENTENCES,
        start=1,
    ):

        print("\n" + "=" * 70)

        print(
            f"TEST {index}/"
            f"{len(TEST_SENTENCES)}"
        )

        print("=" * 70)

        print(
            f"\nEnglish:\n"
            f"{english_text}"
        )


        # ----------------------------------------------------
        # FULL FINE-TUNING
        # ----------------------------------------------------

        print(
            "\nGenerating FULL "
            "Fine-Tuning output..."
        )

        full_output = translate(

            full_model,

            tokenizer,

            english_text,
        )

        print(
            f"\nFULL:\n"
            f"{full_output}"
        )


        # ----------------------------------------------------
        # LORA
        # ----------------------------------------------------

        print(
            "\nGenerating LoRA output..."
        )

        lora_output = translate(

            lora_model,

            tokenizer,

            english_text,
        )

        print(
            f"\nLoRA:\n"
            f"{lora_output}"
        )


        # ----------------------------------------------------
        # QLORA
        # ----------------------------------------------------

        print(
            "\nGenerating QLoRA output..."
        )

        qlora_output = translate(

            qlora_model,

            tokenizer,

            english_text,
        )

        print(
            f"\nQLoRA:\n"
            f"{qlora_output}"
        )


        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        results.append(

            {

                "english": english_text,

                "full_finetuning": full_output,

                "lora": lora_output,

                "qlora": qlora_output,

            }

        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    with open(

        RESULTS_FILE,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            results,

            file,

            indent=4,

            ensure_ascii=False,

        )


    # ========================================================
    # COMPLETED
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "MODEL COMPARISON COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)

    print(
        f"\nResults saved to:\n"
        f"{RESULTS_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()