# Uttar Karnataka Kannada Translation

**English → Kannada → Uttar Karnataka Regional Dialect**

A fine-tuned Neural Machine Translation (NMT) project that translates English sentences into Kannada with a focus on the Uttar Karnataka dialect, particularly the Vijayapura and Bagalkot regions.

The project explores three model adaptation approaches using Meta's NLLB-200 distilled model:
- Full Fine-Tuning
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized Low-Rank Adaptation)

## Project Objectives

- Build an English-to-Uttar Karnataka Kannada translation system.
- Adapt a pretrained multilingual translation model to regional dialect data.
- Compare Full Fine-Tuning, LoRA and QLoRA.
- Develop a complete dataset preparation and evaluation pipeline.
- Provide an interactive translation application with Kannada text-to-speech.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python 3.11 |
| Base Model | NLLB-200 Distilled 600M |
| Deep Learning | PyTorch, Hugging Face Transformers |
| Fine-Tuning | PEFT, LoRA, QLoRA |
| Quantization | 4-bit NF4 |
| Dataset | JSONL |
| Evaluation | BLEU, chrF, Exact Match |
| Application | Streamlit |
| Text-to-Speech | gTTS |

## Model Fine-Tuning Approaches

### 1. Full Fine-Tuning
Updates the pretrained model's trainable parameters using the regional translation dataset.

### 2. LoRA
Keeps the original model weights frozen and trains additional low-rank adapter matrices to reduce the number of trainable parameters.

### 3. QLoRA
Uses 4-bit quantization of the pretrained model along with LoRA adapters to reduce memory requirements during fine-tuning.

## Project Architecture

```text
English Input
     |
     v
Dataset Preparation
     |
     v
Dataset Cleaning & Validation
     |
     v
Train / Evaluation / Test Split
     |
     v
NLLB-200 Distilled 600M
     |
     +---------------------+
     |          |          |
     v          v          v
   Full       LoRA        QLoRA
Fine-Tuning  Fine-Tuning  Fine-Tuning
     |          |          |
     +----------+----------+
                |
                v
       Model Evaluation
                |
                v
       Translation Application
                |
                v
       Kannada Text Output
                |
                v
       Text-to-Speech
```

## Dataset Pipeline

The dataset pipeline includes:

1. Dataset generation and collection.
2. Kannada dialect translation corrections.
3. Duplicate removal and data cleaning.
4. Dataset validation.
5. Train, evaluation and test splitting.

The project uses JSONL records with the following format:

```json
{
  "english": "How are you?",
  "uttar_karnataka": "ನೀ ಹೆಂಗ ಅದಿ?"
}
```

The complete training datasets and downloaded external datasets are excluded from this public repository. A small sample dataset can be added separately where redistribution is permitted.

## Project Structure

```text
uttar-karnataka-kannada-translation/
│
├── fine_tuning/
│   ├── uk_config.py
│   ├── uk_dataset.py
│   ├── uk_train_full.py
│   ├── uk_train_lora.py
│   ├── uk_train_qlora.py
│   ├── uk_inference_full.py
│   ├── uk_inference_lora.py
│   └── uk_inference_qlora.py
│
├── scripts/
│   ├── download_nllb.py
│   ├── prepare_translation_dataset.py
│   ├── uk_clean_dataset.py
│   ├── uk_dataset_generator.py
│   ├── uk_validate_dataset.py
│   ├── uk_split_dataset.py
│   ├── uk_evaluate_full.py
│   ├── uk_evaluate_lora.py
│   ├── uk_evaluate_qlora.py
│   ├── uk_evaluate_all.py
│   └── compare_models.py
│
├── app.py
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/pavanramubhasme/uttar-karnataka-kannada-translation.git

cd uttar-karnataka-kannada-translation
```

Create a virtual environment:

```bash
uv venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
uv pip install -r requirements.txt
```

Ensure PyTorch is installed with a compatible CUDA configuration if GPU acceleration is required.

## Training

Prepare and validate the dataset:

```bash
python scripts/uk_clean_dataset.py
python scripts/uk_validate_dataset.py
python scripts/uk_split_dataset.py
```

Run the different fine-tuning approaches:

**Full Fine-Tuning**

```bash
python -m fine_tuning.uk_train_full
```

**LoRA**

```bash
python -m fine_tuning.uk_train_lora
```

**QLoRA**

```bash
python -m fine_tuning.uk_train_qlora
```

## Model Evaluation

Evaluate all three approaches:

```bash
python scripts/uk_evaluate_all.py
```

Evaluation metrics:

- **BLEU:** Measures translation n-gram overlap.
- **chrF:** Measures character-level translation similarity.
- **Exact Match:** Measures the percentage of predictions exactly matching the reference translations.

## Run the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application provides:
- English-to-Uttar Karnataka Kannada translation.
- Model selection between Full Fine-Tuning, LoRA and QLoRA.
- Translation comparison.
- Translation history.
- Kannada text-to-speech.

**Note:** The text-to-speech component uses standard Kannada speech support and is not a dedicated Uttar Karnataka dialect speech model.

## Future Improvements

- Expand the regional dialect dataset.
- Improve translation quality through native-speaker validation.
- Add automated translation quality evaluation.
- Compare model performance, training time and GPU memory consumption.
- Explore dedicated regional dialect speech generation.
- Deploy the application as a hosted translation service.

## Author

**Pavan Kumar Bhasme**

Embedded Software Engineer | AI/ML & Generative AI Enthusiast

GitHub: https://github.com/pavanramubhasme

---

*This project is an experimental regional-language NLP system focused on adapting multilingual translation models to Uttar Karnataka Kannada.*