import json

from pathlib import Path

import evaluate
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
    / "evaluation"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "uk_qlora_test_results.json"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    samples = []

    with open(
        uk_config.UK_TEST_FILE,
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
# GENERATE TRANSLATION
# ============================================================

def generate_translation(
    model,
    tokenizer,
    english_text,
    device,
):

    inputs = tokenizer(
        english_text,
        return_tensors="pt",
        truncation=True,
        max_length=uk_config.MAX_SOURCE_LENGTH,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    target_language_id = (
        tokenizer.convert_tokens_to_ids(
            uk_config.TARGET_LANGUAGE
        )
    )

    with torch.inference_mode():

        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=target_language_id,
            max_new_tokens=128,
            num_beams=5,
        )

    prediction = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0].strip()

    return prediction


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NLLB → UTTAR KARNATAKA QLoRA TEST EVALUATION")
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
    # CREATE RESULTS DIRECTORY
    # --------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # LOAD METRICS
    # --------------------------------------------------

    print("\nLoading evaluation metrics...")

    bleu_metric = evaluate.load(
        "sacrebleu"
    )

    chrf_metric = evaluate.load(
        "chrf"
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
    # 4-BIT QUANTIZATION CONFIGURATION
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

    print("Loading trained QLoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        str(uk_config.UK_QLORA_OUTPUT_DIR),
    )

    model.eval()

    # --------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------

    print("\nLoading test dataset...")

    examples = load_test_data()

    print(
        f"Test samples: "
        f"{len(examples)}"
    )

    # --------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------

    predictions = []

    references = []

    results = []

    # --------------------------------------------------
    # GENERATE PREDICTIONS
    # --------------------------------------------------

    print("\nGenerating translations...\n")

    for index, example in enumerate(
        examples,
        start=1,
    ):

        english = example["english"]

        reference = example[
            "uttar_karnataka"
        ]

        print(
            f"[{index}/{len(examples)}] "
            f"Translating..."
        )

        prediction = generate_translation(
            model,
            tokenizer,
            english,
            device,
        )

        predictions.append(prediction)

        # sacreBLEU expects a list of references
        references.append([reference])

        results.append(
            {
                "english": english,

                "reference_uttar_karnataka": (
                    reference
                ),

                "predicted_uttar_karnataka": (
                    prediction
                ),
            }
        )

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    print("\nCalculating BLEU score...")

    bleu_result = bleu_metric.compute(
        predictions=predictions,
        references=references,
    )

    print("Calculating chrF score...")

    chrf_result = chrf_metric.compute(
        predictions=predictions,
        references=references,
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    final_results = {

        "model": (
            "NLLB-200-distilled-600M + QLoRA 4-bit"
        ),

        "metrics": {

            "bleu": bleu_result["score"],

            "chrf": chrf_result["score"],
        },

        "total_test_samples": len(
            examples
        ),

        "results": results,
    }

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # DISPLAY FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("QLoRA TEST EVALUATION COMPLETED")
    print("=" * 60)

    print(
        f"\nTotal Test Samples: "
        f"{len(examples)}"
    )

    print(
        f"\nBLEU Score: "
        f"{bleu_result['score']:.4f}"
    )

    print(
        f"chrF Score: "
        f"{chrf_result['score']:.4f}"
    )

    print(
        f"\nResults saved to:\n"
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()