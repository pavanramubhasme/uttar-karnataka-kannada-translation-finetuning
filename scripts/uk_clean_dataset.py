import json

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "data/uk_raw/uk_seed.jsonl"
)

OUTPUT_DIR = Path(
    "data/uk_cleaned"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "uk_cleaned.jsonl"
)


# ============================================================
# SAFE WORD REPLACEMENTS
# ============================================================

REPLACEMENTS = {

    # Preferred Uttar Karnataka words
    "ಹತ್ರ": "ಕಡೆ",

    "ನಂತರ": "ಆಮೇಲೆ",

    "ಸಾಕಷ್ಟು": "ಭಾಳ",

    "ತರಕಾರಿ": "ಕಾಯಿಪಲ್ಲೆ",

    "ರಜೆ": "ಸೂಟಿ",

    "ಹೊಗಬೇಡ": "ಹೋಗ್ಬ್ಯಾಡ",

    "ಚಿಲ್ಲರೆ": "ಚಿಲ್ಲರ",
}


# ============================================================
# PREFERRED TRANSLATIONS FOR CONFLICTING SENTENCES
# ============================================================

PREFERRED_TRANSLATIONS = {

    "how are you?":
        "ನೀ ಹೆಂಗ ಅದಿ?",

    "where are you going?":
        "ನೀ ಎಲ್ಲಿ ಹೊಂಟಿ?",

    "wait for me.":
        "ನನ್ನ ಸಮಂದ್ ನಿಂದರ್",

    "please help me":
        "ನಂಗ್ ಸ್ವಲ್ಪ ಸಹಾಯ ಮಾಡ್ಪಾ.",

    "what is your name?":
        "ನಿನ್ನ ಹೆಸರ್ ಏನ್?",

    "what do you want?":
        "ನಿನಗ್ ಏನ್ ಬೇಕ್?",

    "give me some water":
        "ನಂಗ್ ನೀರ್ ಕೊಡ.",

    "i don't know":
        "ನನಗ್ ಗೊತ್ತಿಲ್ಲ",

    "what time will you come?":
        "ನೀ ಎಷ್ಟೊತ್ತಿಗೆ ಬರ್ತೀ?",

    "what are you saying?":
        "ನೀ ಏನ್ ಹೇಳಾತಿ?",

    "are you ready?":
        "ನೀ ರೆಡಿ ಅದಿ?",

    "what are you eating?":
        "ನೀ ಏನ್ ತಿನ್ನಾತಿ?",

    "where is the bus stop?":
        "ಬಸ್ ಸ್ಟಾಪ್ ಎಲ್ಲಿ ಐತಿ?",

    "when will the bus come?":
        "ಬಸ್ ಯಾವಾಗ ಬರ್ತೀತಿ?",

    "don't tell anyone.":
        "ಯಾರಿಗೂ ಹೇಳ್ಬ್ಯಾಡ.",

    "don't come late.":
        "ಲೇಟ್ ಬರಬ್ಯಾಡ.",

    "let's go now.":
        "ಬಾ, ಈಗ ಹೊಗೋಣ.",

    "can you help me?":
        "ನೀ ನನಗ್ ಸಹಾಯ ಮಾಡ್ತಿಯಾ?",

    "don't worry about it.":
        "ಅದ್ರ ಬಗ್ಗೆ ಚಿಂತೆ ಮಾಡ್ಬ್ಯಾಡ.",

    "don't wait for me.":
        "ನನ್ನ ಸಮಂದ್ ನಿಂದರ್ಬ್ಯಾಡ್.",

    "i will be there soon.":
        "ನಾ ಬೇಗ ಅಲ್ಲಿ ಇರ್ತೀನಿ.",

    "what did you eat?":
        "ನೀ ಏನ್ ತಿಂದಿ?",

    "check your pocket.":
        "ನಿನ್ನ ಪಾಕೆಟ್ ಚೆಕ್ ಮಾಡ್.",

    "why are you angry?":
        "ನೀ ಯಾಕ್ ಕೋಪ ಮಾಡ್ಕೊಂಡಿ?",

    "if he had scored 100, it would have been good.":
        "ಅವ 100 ಹೊಡದಿದ್ದಂದ್ರ ಚಲೋ ಇರತಿತ್ತ.",

    "wait for me. i am coming.":
        "ನನಗ್ ಕಾಯ್. ನಾ ಬರ್ತೀನಿ.",

    "do you have a plan?":
        "ನಿನ್ನ ಕಡೆ ಪ್ಲಾನ್ ಐತಿ?",
}


# ============================================================
# PERSON FORM CORRECTIONS
# ============================================================

PERSON_FORM_REPLACEMENTS = {

    "ನೀ ರೆಡಿ ಅದಿಯಾ?":
        "ನೀ ರೆಡಿ ಅದಿ?",

    "ನೀ ಚಲೋ ಅದಿಯಾ?":
        "ನೀ ಚಲೋ ಅದಿ?",

    "ನೀ ಈಗ ಫ್ರೀ ಅದಿಯಾ?":
        "ನೀ ಈಗ ಫ್ರೀ ಅದಿ?",

    "ನೀ ಮನಿಯಾಗ ಐತಿಯಾ?":
        "ನೀ ಮನಿಯಾಗ ಆದಿ?",

    "ನೀ ರೆಡಿ ಐತಿಯಾ?":
        "ನೀ ರೆಡಿ ಅದಿ?",

    "ನೀ ಚಲೋ ಐತಿಯಾ?":
        "ನೀ ಚಲೋ ಅದಿ?",

    "ಈಗ ಫ್ರೀ ಐತಿಯಾ?":
        "ಈಗ ಫ್ರೀ ಅದಿ?",

    "ನೀ ಈಗ ಮನಿಯಾಗ ಐತಿಯಾ?":
        "ನೀ ಈಗ ಮನಿಯಾಗ ಆದಿ?",

    "ಹೊಗಕ್ ರೆಡಿ ಐತಿಯಾ?":
        "ಹೊಗಕ್ ರೆಡಿ ಅದಿ?",
}


# ============================================================
# SPECIFIC PERMANENT SENTENCE CORRECTIONS
# ============================================================

SPECIFIC_TRANSLATION_CORRECTIONS = {

    # Bring examples
    "bring this":
        "ಇದನ್ನ ತಗೊಂದು ಬಾ",

    "bring it here":
        "ಅದನ್ನ ಇಲ್ಲಿ ತಗೊಂದು ಬಾ",

    "bring my phone":
        "ನನ್ನ ಫೋನ್ ತಗೊಂದು ಬಾ",

    "bring some water":
        "ಸ್ವಲ್ಪ ನೀರು ತಗೊಂದು ಬಾ",

    "bring the phone. don't forget it.":
        "ಫೋನ್ ತಗೊಂದು ಬಾ. ಅದನ್ನ ಮರಿಬೇಡ.",

    # Sleep / person form example
    "did you sleep well? are you feeling better now?":
        "ನೀ ಚಲೋ ಮಲಗಿದ್ದಿ? ಈಗ ಚಲೋ ಅದಿ?",
}


# ============================================================
# APPLY WORD REPLACEMENTS
# ============================================================

def apply_replacements(text):

    for old_word, new_word in REPLACEMENTS.items():

        text = text.replace(
            old_word,
            new_word,
        )

    return text


# ============================================================
# APPLY PERSON FORM CORRECTIONS
# ============================================================

def apply_person_form_corrections(text):

    for old_text, new_text in (
        PERSON_FORM_REPLACEMENTS.items()
    ):

        if old_text in text:

            text = text.replace(
                old_text,
                new_text,
            )

    return text


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("UK DATASET CLEANING")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_records = []

    seen_records = set()

    removed_duplicates = 0

    modified_records = 0

    # --------------------------------------------------------
    # READ INPUT DATASET
    # --------------------------------------------------------

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            english = record[
                "english"
            ].strip()

            dialect = record[
                "uttar_karnataka"
            ].strip()

            original_dialect = dialect

            english_key = english.lower()

            # ------------------------------------------------
            # APPLY PREFERRED CONFLICT TRANSLATION
            # ------------------------------------------------

            if english_key in PREFERRED_TRANSLATIONS:

                dialect = PREFERRED_TRANSLATIONS[
                    english_key
                ]

            # ------------------------------------------------
            # APPLY SPECIFIC PERMANENT CORRECTIONS
            # ------------------------------------------------

            if english_key in SPECIFIC_TRANSLATION_CORRECTIONS:

                dialect = SPECIFIC_TRANSLATION_CORRECTIONS[
                    english_key
                ]

            # ------------------------------------------------
            # APPLY PERSON FORM CORRECTIONS
            # ------------------------------------------------

            dialect = apply_person_form_corrections(
                dialect
            )

            # ------------------------------------------------
            # APPLY SAFE WORD REPLACEMENTS
            # ------------------------------------------------

            dialect = apply_replacements(
                dialect
            )

            # ------------------------------------------------
            # TRACK MODIFICATIONS
            # ------------------------------------------------

            if dialect != original_dialect:

                modified_records += 1

            # ------------------------------------------------
            # REMOVE EXACT DUPLICATES
            # ------------------------------------------------

            record_key = (
                english.lower(),
                dialect,
            )

            if record_key in seen_records:

                removed_duplicates += 1

                continue

            seen_records.add(
                record_key
            )

            # ------------------------------------------------
            # STORE CLEANED RECORD
            # ------------------------------------------------

            cleaned_records.append(
                {
                    "english": english,
                    "uttar_karnataka": dialect,
                }
            )

    # ========================================================
    # SAVE CLEANED DATASET
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        for record in cleaned_records:

            json.dump(
                record,
                file,
                ensure_ascii=False,
            )

            file.write("\n")

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print(
        f"\nOriginal dataset: "
        f"{INPUT_FILE}"
    )

    print(
        f"\nCleaned dataset: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nFinal samples: "
        f"{len(cleaned_records)}"
    )

    print(
        f"Removed duplicates: "
        f"{removed_duplicates}"
    )

    print(
        f"Modified records: "
        f"{modified_records}"
    )

    print("\n" + "=" * 60)
    print("DATASET CLEANING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()