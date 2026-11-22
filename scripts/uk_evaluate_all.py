
import gc
import json
import hashlib
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
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_FILE = PROJECT_ROOT / "data" / "uk_dataset" / "test.jsonl"

RESULTS_DIR = PROJECT_ROOT / "outputs" / "evaluation"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_NEW_TOKENS = uk_config.MAX_TARGET_LENGTH
NUM_BEAMS = 5


# ============================================================
# LOAD AND VALIDATE TEST DATA
# ============================================================

def load_test_data():

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test file not found: {TEST_FILE}"
        )

    samples = []

    with open(TEST_FILE, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            sample = json.loads(line)

            if (
                not sample.get("english")
                or not sample.get("uttar_karnataka")
            ):
                raise ValueError(
                    f"Missing English or translation at line {line_number}"
                )

            samples.append(sample)

    if not samples:
        raise ValueError("Test dataset is empty.")

    english_sentences = [
        item["english"].strip()
        for item in samples
    ]

    duplicates = (
        len(english_sentences)
        - len(set(english_sentences))
    )

    print("\nTEST DATASET INFORMATION")
    print("-" * 50)

    print(f"Test file: {TEST_FILE}")
    print(f"Total samples: {len(samples)}")
    print(f"Unique English sentences: {len(set(english_sentences))}")
    print(f"Duplicate English entries: {duplicates}")

    if duplicates:
        print(
            "WARNING: Duplicate entries are retained "
            "for this comparison."
        )

    return samples


# ============================================================
# GENERATE TRANSLATION
# ============================================================

def generate_translation(model, tokenizer, english_text):

    inputs = tokenizer(
        english_text,
        return_tensors="pt",
        truncation=True,
        max_length=uk_config.MAX_SOURCE_LENGTH,
    )

    input_device = model.get_input_embeddings().weight.device

    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    target_language_id = tokenizer.convert_tokens_to_ids(
        uk_config.TARGET_LANGUAGE
    )

    with torch.inference_mode():

        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=target_language_id,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            do_sample=False,
        )

    prediction = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0].strip()

    return prediction


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model_name,
    model_type,
    examples,
    bleu_metric,
    chrf_metric,
):

    print("\n" + "=" * 65)
    print(f"EVALUATING: {model_name}")
    print("=" * 65)

    tokenizer_path = {
        "full": uk_config.UK_FULL_OUTPUT_DIR,
        "lora": uk_config.UK_LORA_OUTPUT_DIR,
        "qlora": uk_config.UK_QLORA_OUTPUT_DIR,
    }[model_type]

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        src_lang=uk_config.SOURCE_LANGUAGE,
        tgt_lang=uk_config.TARGET_LANGUAGE,
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    base_model = None

    if model_type == "full":

        print("Loading Full Fine-Tuned Model...")

        model = AutoModelForSeq2SeqLM.from_pretrained(
            uk_config.UK_FULL_OUTPUT_DIR,
            torch_dtype=torch.float32,
        )

        model = model.to("cuda")

    elif model_type == "lora":

        print("Loading LoRA Base Model...")

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            uk_config.MODEL_NAME,
            torch_dtype=torch.float32,
        )

        base_model = base_model.to("cuda")

        print("Loading LoRA Adapter...")

        model = PeftModel.from_pretrained(
            base_model,
            str(uk_config.UK_LORA_OUTPUT_DIR),
        )

        model = model.to("cuda")

    elif model_type == "qlora":

        print("Configuring 4-bit Quantization...")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=uk_config.QLORA_QUANT_TYPE,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=(
                uk_config.QLORA_USE_DOUBLE_QUANT
            ),
        )

        print("Loading QLoRA Base Model in 4-bit...")

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            uk_config.MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
        )

        print("Loading QLoRA Adapter...")

        model = PeftModel.from_pretrained(
            base_model,
            str(uk_config.UK_QLORA_OUTPUT_DIR),
        )

    else:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    model.eval()

    # --------------------------------------------------------
    # GENERATE PREDICTIONS
    # --------------------------------------------------------

    predictions = []
    references = []
    results = []

    print("\nGenerating translations...\n")

    for index, example in enumerate(examples, start=1):

        english = example["english"].strip()

        reference = example["uttar_karnataka"].strip()

        print(
            f"[{index}/{len(examples)}] "
            f"Translating..."
        )

        prediction = generate_translation(
            model,
            tokenizer,
            english,
        )

        predictions.append(prediction)

        references.append([reference])

        results.append({
            "english": english,
            "reference_uttar_karnataka": reference,
            "predicted_uttar_karnataka": prediction,
        })

    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    print("\nCalculating BLEU...")

    bleu_result = bleu_metric.compute(
        predictions=predictions,
        references=references,
    )

    print("Calculating chrF...")

    chrf_result = chrf_metric.compute(
        predictions=predictions,
        references=references,
    )

    exact_matches = sum(
        prediction.strip() == reference[0].strip()
        for prediction, reference in zip(
            predictions,
            references,
        )
    )

    exact_match_percentage = (
        exact_matches / len(examples)
    ) * 100

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    final_results = {

        "model": model_name,

        "model_type": model_type,

        "total_test_samples": len(examples),

        "metrics": {

            "bleu": bleu_result["score"],

            "chrf": chrf_result["score"],

            "exact_matches": exact_matches,

            "exact_match_percentage": exact_match_percentage,
        },

        "results": results,
    }

    output_file = (
        RESULTS_DIR
        / f"uk_{model_type}_test_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # DISPLAY MODEL RESULTS
    # --------------------------------------------------------

    print("\n" + "-" * 55)

    print(f"MODEL: {model_name}")

    print(f"Test Samples: {len(examples)}")

    print(f"BLEU: {bleu_result['score']:.4f}")

    print(f"chrF: {chrf_result['score']:.4f}")

    print(f"Exact Matches: {exact_matches}")

    print(f"Exact Match: {exact_match_percentage:.2f}%")

    print(f"Saved: {output_file}")

    print("-" * 55)

    # --------------------------------------------------------
    # RELEASE GPU MEMORY
    # --------------------------------------------------------

    del model
    del tokenizer

    if base_model is not None:
        del base_model

    gc.collect()

    torch.cuda.empty_cache()

    return final_results


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 75)
    print("COMMON UTTAR KARNATAKA MODEL EVALUATION")
    print("=" * 75)

    # --------------------------------------------------------
    # GPU CHECK
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )

    print(
        f"\nGPU: {torch.cuda.get_device_name(0)}"
    )

    # --------------------------------------------------------
    # LOAD COMMON TEST DATASET
    # --------------------------------------------------------

    examples = load_test_data()

    # --------------------------------------------------------
    # DATASET FINGERPRINT
    # --------------------------------------------------------

    dataset_hash = hashlib.sha256(
        TEST_FILE.read_bytes()
    ).hexdigest()

    print(f"\nDataset SHA256: {dataset_hash}")

    # --------------------------------------------------------
    # LOAD EVALUATION METRICS
    # --------------------------------------------------------

    print("\nLoading evaluation metrics...")

    bleu_metric = evaluate.load("sacrebleu")

    chrf_metric = evaluate.load("chrf")

    # --------------------------------------------------------
    # MODEL CONFIGURATION
    # --------------------------------------------------------

    models = [

        (
            "FULL FINE-TUNING",
            "full",
        ),

        (
            "LoRA",
            "lora",
        ),

        (
            "QLoRA (4-bit)",
            "qlora",
        ),
    ]

    all_results = []

    # --------------------------------------------------------
    # EVALUATE ALL THREE MODELS
    # --------------------------------------------------------

    for model_name, model_type in models:

        result = evaluate_model(
            model_name=model_name,
            model_type=model_type,
            examples=examples,
            bleu_metric=bleu_metric,
            chrf_metric=chrf_metric,
        )

        if result["total_test_samples"] != len(examples):

            raise RuntimeError(
                f"{model_name} evaluated an incorrect sample count."
            )

        all_results.append(result)

    # --------------------------------------------------------
    # CREATE FINAL COMPARISON REPORT
    # --------------------------------------------------------

    comparison = {

        "test_dataset": str(TEST_FILE),

        "dataset_sha256": dataset_hash,

        "total_test_samples": len(examples),

        "generation_settings": {

            "max_source_length": uk_config.MAX_SOURCE_LENGTH,

            "max_new_tokens": MAX_NEW_TOKENS,

            "num_beams": NUM_BEAMS,

            "do_sample": False,
        },

        "models": [],
    }

    for result in all_results:

        comparison["models"].append({

            "model": result["model"],

            "model_type": result["model_type"],

            "total_test_samples": result["total_test_samples"],

            "bleu": result["metrics"]["bleu"],

            "chrf": result["metrics"]["chrf"],

            "exact_matches": result["metrics"]["exact_matches"],

            "exact_match_percentage": (
                result["metrics"]["exact_match_percentage"]
            ),
        })

    comparison_file = (
        RESULTS_DIR / "uk_model_comparison.json"
    )

    with open(
        comparison_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            comparison,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # DISPLAY FINAL COMPARISON
    # --------------------------------------------------------

    print("\n" + "=" * 80)

    print("FINAL MODEL COMPARISON")

    print("=" * 80)

    print(
        f"\n{'MODEL':<25}"
        f"{'BLEU':>12}"
        f"{'chrF':>12}"
        f"{'EXACT MATCH':>18}"
        f"{'SAMPLES':>12}"
    )

    print("-" * 80)

    for result in comparison["models"]:

        print(
            f"{result['model']:<25}"
            f"{result['bleu']:>12.4f}"
            f"{result['chrf']:>12.4f}"
            f"{result['exact_match_percentage']:>17.2f}%"
            f"{result['total_test_samples']:>12}"
        )

    print("-" * 80)

    print(
        f"\nCommon test samples per model: {len(examples)}"
    )

    print(
        f"Dataset SHA256: {dataset_hash}"
    )

    print(
        f"\nComparison saved to:\n{comparison_file}"
    )

    print("\nALL MODEL EVALUATIONS COMPLETED.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
