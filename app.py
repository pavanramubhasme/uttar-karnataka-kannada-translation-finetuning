
import gc
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch

from peft import PeftModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from fine_tuning.uk_config import uk_config


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Uttar Karnataka Translator",
    page_icon="🗣️",
    layout="wide",
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_OPTIONS = {
    "Full Fine-tuned": "full",
    "LoRA": "lora",
    "QLoRA": "qlora",
}

MODEL_PATHS = {
    "full": uk_config.UK_FULL_OUTPUT_DIR,
    "lora": uk_config.UK_LORA_OUTPUT_DIR,
    "qlora": uk_config.UK_QLORA_OUTPUT_DIR,
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(model_key):

    model_path = Path(MODEL_PATHS[model_key])

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}"
        )

    if model_key == "qlora" and not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA requires a CUDA GPU."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        src_lang=uk_config.SOURCE_LANGUAGE,
        tgt_lang=uk_config.TARGET_LANGUAGE,
    )

    # --------------------------------------------------------
    # FULL FINE-TUNED MODEL
    # --------------------------------------------------------

    if model_key == "full":

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float32,
        )

        model = model.to(device)

    # --------------------------------------------------------
    # LoRA MODEL
    # --------------------------------------------------------

    elif model_key == "lora":

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            uk_config.MODEL_NAME,
            torch_dtype=torch.float32,
        )

        base_model = base_model.to(device)

        model = PeftModel.from_pretrained(
            base_model,
            str(model_path),
        )

        model = model.to(device)

    # --------------------------------------------------------
    # QLoRA MODEL
    # --------------------------------------------------------

    elif model_key == "qlora":

        quantization_config = BitsAndBytesConfig(

            load_in_4bit=uk_config.QLORA_LOAD_IN_4BIT,

            bnb_4bit_quant_type=(
                uk_config.QLORA_QUANT_TYPE
            ),

            bnb_4bit_compute_dtype=torch.float16,

            bnb_4bit_use_double_quant=(
                uk_config.QLORA_USE_DOUBLE_QUANT
            ),
        )

        base_model = AutoModelForSeq2SeqLM.from_pretrained(

            uk_config.MODEL_NAME,

            quantization_config=quantization_config,

            device_map="auto",
        )

        model = PeftModel.from_pretrained(
            base_model,
            str(model_path),
        )

    else:
        raise ValueError("Invalid model selection")

    model.eval()

    return model, tokenizer


# ============================================================
# TRANSLATION FUNCTION
# ============================================================

def translate_text(text, model, tokenizer):

    inputs = tokenizer(
        text,

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

            max_new_tokens=uk_config.MAX_TARGET_LENGTH,

            num_beams=5,
        )

    translation = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0].strip()

    return translation


# ============================================================
# CLEAR MODEL MEMORY
# ============================================================

def clear_model_memory():

    load_model.clear()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# TEXT TO SPEECH
# ============================================================

def generate_speech(text):

    try:

        from gtts import gTTS

        audio_buffer = BytesIO()

        gTTS(
            text=text,
            lang="kn",
        ).write_to_fp(audio_buffer)

        audio_buffer.seek(0)

        return audio_buffer

    except Exception as error:

        st.error(
            f"Speech generation failed: {error}"
        )

        return None


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "active_model" not in st.session_state:
    st.session_state.active_model = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Settings")

selected_model_label = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_OPTIONS.keys()),
)

selected_model_key = MODEL_OPTIONS[
    selected_model_label
]

st.sidebar.caption(
    "Only the selected model is loaded for normal translation."
)

if torch.cuda.is_available():

    st.sidebar.success(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

else:

    st.sidebar.info(
        "CUDA not detected. Full and LoRA will use CPU. "
        "QLoRA requires CUDA."
    )


if st.sidebar.button(
    "Clear Translation History",
    use_container_width=True,
):

    st.session_state.history = []

    st.session_state.last_result = None

    st.rerun()


# ============================================================
# MAIN APPLICATION
# ============================================================

st.title("🗣️ English → Uttar Karnataka Kannada")

st.write(
    "Translate English sentences into Uttar Karnataka Kannada dialect."
)

st.caption(
    f"Selected Model: {selected_model_label}"
)


# ============================================================
# INPUT
# ============================================================

english_text = st.text_area(

    "Enter English Text",

    placeholder="Example: Where are you going now?",

    height=140,
)


translate_col, compare_col = st.columns(2)


with translate_col:

    translate_clicked = st.button(
        "Translate",
        type="primary",
        use_container_width=True,
    )


with compare_col:

    compare_clicked = st.button(
        "Compare All 3 Models",
        use_container_width=True,
    )


# ============================================================
# SINGLE MODEL TRANSLATION
# ============================================================

if translate_clicked:

    if not english_text.strip():

        st.warning("Please enter English text.")

    else:

        try:

            if st.session_state.active_model != selected_model_key:

                clear_model_memory()

                st.session_state.active_model = selected_model_key

            with st.spinner(
                f"Loading {selected_model_label} and translating..."
            ):

                model, tokenizer = load_model(
                    selected_model_key
                )

                translated_text = translate_text(
                    english_text.strip(),
                    model,
                    tokenizer,
                )

            result = {

                "english": english_text.strip(),

                "translation": translated_text,

                "model": selected_model_label,
            }

            st.session_state.last_result = result

            st.session_state.history.insert(
                0,
                result.copy(),
            )

        except Exception as error:

            st.error(
                f"Translation failed: {error}"
            )


# ============================================================
# DISPLAY TRANSLATION
# ============================================================

if st.session_state.last_result:

    result = st.session_state.last_result

    st.divider()

    st.subheader("Uttar Karnataka Translation")

    st.markdown(
        f"**English:** {result['english']}"
    )

    st.markdown(
        f"**Model:** {result['model']}"
    )

    st.text_area(
        "Kannada Output",
        value=result["translation"],
        height=110,
        key="single_translation_output",
    )

    if st.button("🔊 Generate Kannada Speech"):

        audio_data = generate_speech(
            result["translation"]
        )

        if audio_data:

            st.audio(
                audio_data,
                format="audio/mp3",
            )

    st.caption(
        "Use the copy icon in the output text area to copy the translation."
    )


# ============================================================
# COMPARE ALL THREE MODELS
# ============================================================

if compare_clicked:

    if not english_text.strip():

        st.warning(
            "Enter English text before comparing models."
        )

    else:

        results = {}

        errors = {}

        st.divider()

        st.subheader("Model Comparison")

        progress = st.progress(0)

        for index, (model_label, model_key) in enumerate(
            MODEL_OPTIONS.items()
        ):

            try:

                with st.spinner(
                    f"Running {model_label}..."
                ):

                    clear_model_memory()

                    st.session_state.active_model = None

                    model, tokenizer = load_model(
                        model_key
                    )

                    output = translate_text(
                        english_text.strip(),
                        model,
                        tokenizer,
                    )

                    results[model_label] = output

                    del model, tokenizer

                    clear_model_memory()

            except Exception as error:

                errors[model_label] = str(error)

                clear_model_memory()

            progress.progress(
                (index + 1) / len(MODEL_OPTIONS)
            )

        for model_label in MODEL_OPTIONS:

            st.markdown(
                f"**{model_label}**"
            )

            if model_label in results:

                st.text_area(
                    f"{model_label} Output",

                    value=results[model_label],

                    height=90,

                    key=f"comparison_{model_label}",
                )

            else:

                st.error(
                    f"{model_label} failed: "
                    f"{errors.get(model_label, 'Unknown error')}"
                )

        if results:

            history_result = {

                "english": english_text.strip(),

                "translation": " | ".join(
                    f"{name}: {value}"
                    for name, value in results.items()
                ),

                "model": "Model Comparison",
            }

            st.session_state.history.insert(
                0,
                history_result,
            )


# ============================================================
# TRANSLATION HISTORY
# ============================================================

st.divider()

st.subheader("🕘 Translation History")

if not st.session_state.history:

    st.caption(
        "Your translations will appear here."
    )

else:

    for item in st.session_state.history:

        with st.expander(
            f"{item['model']} — {item['english'][:65]}"
        ):

            st.markdown(
                f"**English:** {item['english']}"
            )

            st.markdown(
                "**Uttar Karnataka Output:**"
            )

            st.write(
                item["translation"]
            )