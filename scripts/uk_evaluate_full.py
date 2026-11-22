import json
from pathlib import Path

import evaluate
import torch

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
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
    / "uk_full_test_results.json"
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
            max_new_tokens=uk_config.MAX_TARGET_LENGTH,
            num_beams=5,
            do_sample=False,
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
    print("ENGLISH → UTTAR KARNATAKA")
    print("FULL FINE-TUNING TEST EVALUATION")
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

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        uk_config.UK_FULL_OUTPUT_DIR,
        src_lang=uk_config.SOURCE_LANGUAGE,
        tgt_lang=uk_config.TARGET_LANGUAGE,
    )

    # --------------------------------------------------
    # LOAD FULL FINE-TUNED MODEL
    # --------------------------------------------------

    print(
        "Loading full fine-tuned model..."
    )

    model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            uk_config.UK_FULL_OUTPUT_DIR,
            torch_dtype=torch.float32,
        )
    )

    model = model.to(device)

    model.eval()

    # --------------------------------------------------
    # LOAD TEST DATASET
    # --------------------------------------------------

    print("\nLoading test dataset...")

    examples = load_test_data()

    print(
        f"Test samples: "
        f"{len(examples)}"
    )

    if len(examples) == 0:
        raise ValueError(
            "Test dataset is empty."
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
            model=model,
            tokenizer=tokenizer,
            english_text=english,
            device=device,
        )

        predictions.append(prediction)

        # sacreBLEU expects:
        # [[reference_1], [reference_2], ...]
        references.append(
            [reference]
        )

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

    # ====================================================
    # CALCULATE METRICS
    # ====================================================

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

    # ====================================================
    # FINAL RESULTS
    # ====================================================

    final_results = {

        "model": (
            "NLLB-200-Distilled-600M "
            "Full Fine-Tuning"
        ),

        "metrics": {

            "bleu": bleu_result[
                "score"
            ],

            "chrf": chrf_result[
                "score"
            ],
        },

        "total_test_samples": len(
            examples
        ),

        "results": results,
    }

    # ====================================================
    # SAVE RESULTS
    # ====================================================

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

    # ====================================================
    # DISPLAY RESULTS
    # ====================================================

    print("\n" + "=" * 60)
    print("TEST EVALUATION COMPLETED")
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