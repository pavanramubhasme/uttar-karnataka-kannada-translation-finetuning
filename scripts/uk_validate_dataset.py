import json

from collections import defaultdict
from pathlib import Path


# ============================================================
# INPUT FILE
# ============================================================

INPUT_FILE = Path(
    "data/uk_raw/uk_seed.jsonl"
)


# ============================================================
# PREFERRED DIALECT RULES
# ============================================================

FLAGGED_WORDS = {

    "ಹತ್ರ": "Preferred: ಕಡೆ",

    "ನಂತರ": "Preferred: ಆಮೇಲೆ",

    "ಸಾಕಷ್ಟು": "Preferred: ಭಾಳ",

    "ತರಕಾರಿ": "Preferred: ಕಾಯಿಪಲ್ಲೆ",

    "ರಜೆ": "Preferred: ಸೂಟಿ",

    "ಹೊಗಬೇಡ": "Preferred: ಹೋಗ್ಬ್ಯಾಡ",

    "ಚಿಲ್ಲರೆ": "Preferred: ಚಿಲ್ಲರ",

    "ಅದಿಯಾ": "Review person form",

    "ಐತಿಯಾ": "Review person/object context",
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("UK DATASET VALIDATION")
    print("=" * 60)

    valid_records = []
    errors = []

    exact_duplicates = []

    english_to_translations = defaultdict(list)

    flagged_words = []

    seen_records = set()

    # --------------------------------------------------------
    # READ DATASET
    # --------------------------------------------------------

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            # ------------------------------------------------
            # JSON VALIDATION
            # ------------------------------------------------

            try:

                record = json.loads(line)

            except json.JSONDecodeError as error:

                errors.append(
                    f"Line {line_number}: "
                    f"Invalid JSON - {error}"
                )

                continue

            # ------------------------------------------------
            # FIELD VALIDATION
            # ------------------------------------------------

            english = record.get(
                "english",
                "",
            ).strip()

            dialect = record.get(
                "uttar_karnataka",
                "",
            ).strip()

            if not english:

                errors.append(
                    f"Line {line_number}: "
                    f"Empty English"
                )

                continue

            if not dialect:

                errors.append(
                    f"Line {line_number}: "
                    f"Empty Uttar Karnataka text"
                )

                continue

            # ------------------------------------------------
            # EXACT DUPLICATE CHECK
            # ------------------------------------------------

            record_key = (
                english.lower(),
                dialect,
            )

            if record_key in seen_records:

                exact_duplicates.append(
                    {
                        "line": line_number,
                        "english": english,
                        "uttar_karnataka": dialect,
                    }
                )

            else:

                seen_records.add(
                    record_key
                )

            # ------------------------------------------------
            # SAME ENGLISH → MULTIPLE TRANSLATIONS
            # ------------------------------------------------

            english_to_translations[
                english.lower()
            ].append(
                {
                    "line": line_number,
                    "translation": dialect,
                }
            )

            # ------------------------------------------------
            # FLAG DIALECT WORDS
            # ------------------------------------------------

            for word, message in (
                FLAGGED_WORDS.items()
            ):

                if word in dialect:

                    flagged_words.append(
                        {
                            "line": line_number,
                            "english": english,
                            "translation": dialect,
                            "flagged_word": word,
                            "suggestion": message,
                        }
                    )

            # ------------------------------------------------
            # ADD VALID RECORD
            # ------------------------------------------------

            valid_records.append(
                record
            )

    # ========================================================
    # FIND CONFLICTING TRANSLATIONS
    # ========================================================

    conflicts = []

    for english, translations in (
        english_to_translations.items()
    ):

        unique_translations = set(
            item["translation"]
            for item in translations
        )

        if len(unique_translations) > 1:

            conflicts.append(
                {
                    "english": english,
                    "translations": translations,
                }
            )

    # ========================================================
    # DISPLAY SUMMARY
    # ========================================================

    print(
        f"\nValid samples: "
        f"{len(valid_records)}"
    )

    print(
        f"Errors: "
        f"{len(errors)}"
    )

    print(
        f"Exact duplicates: "
        f"{len(exact_duplicates)}"
    )

    print(
        f"Conflicting English sentences: "
        f"{len(conflicts)}"
    )

    print(
        f"Flagged dialect words: "
        f"{len(flagged_words)}"
    )

    # ========================================================
    # DISPLAY ERRORS
    # ========================================================

    if errors:

        print("\n" + "=" * 60)
        print("ERRORS")
        print("=" * 60)

        for error in errors:

            print(error)

    # ========================================================
    # DISPLAY DUPLICATES
    # ========================================================

    if exact_duplicates:

        print("\n" + "=" * 60)
        print("EXACT DUPLICATES")
        print("=" * 60)

        for item in exact_duplicates:

            print(
                f"\nLine: {item['line']}"
            )

            print(
                f"English: "
                f"{item['english']}"
            )

            print(
                f"UK: "
                f"{item['uttar_karnataka']}"
            )

    # ========================================================
    # DISPLAY CONFLICTS
    # ========================================================

    if conflicts:

        print("\n" + "=" * 60)
        print("CONFLICTING TRANSLATIONS")
        print("=" * 60)

        for conflict in conflicts:

            print(
                f"\nEnglish: "
                f"{conflict['english']}"
            )

            unique_items = set()

            for item in (
                conflict["translations"]
            ):

                text = item["translation"]

                if text in unique_items:
                    continue

                unique_items.add(text)

                print(
                    f"  Line {item['line']}: "
                    f"{text}"
                )

    # ========================================================
    # DISPLAY FLAGGED WORDS
    # ========================================================

    if flagged_words:

        print("\n" + "=" * 60)
        print("FLAGGED DIALECT WORDS")
        print("=" * 60)

        for item in flagged_words:

            print(
                f"\nLine: "
                f"{item['line']}"
            )

            print(
                f"English: "
                f"{item['english']}"
            )

            print(
                f"UK: "
                f"{item['translation']}"
            )

            print(
                f"Found: "
                f"{item['flagged_word']}"
            )

            print(
                f"Suggestion: "
                f"{item['suggestion']}"
            )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n" + "=" * 60)

    if not errors:

        print(
            "DATASET FORMAT VALIDATION SUCCESSFUL ✓"
        )

    else:

        print(
            "DATASET HAS FORMAT ERRORS ✗"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()