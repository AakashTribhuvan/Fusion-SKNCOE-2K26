from __future__ import annotations

import re
from typing import Any


def normalize_phrase(value: str | None) -> str:
    if value is None:
        return ""
    phrase = value.strip().lower()
    phrase = phrase.replace("-", " ")
    phrase = re.sub(r"[.,!?;:]", " ", phrase)
    phrase = re.sub(r"\s+", " ", phrase)
    return phrase.strip()


class PhraseMatchService:
    @staticmethod
    def compare(expected_phrase: str, recognized_phrase: str | None) -> dict[str, Any]:
        expected = normalize_phrase(expected_phrase)
        recognized = normalize_phrase(recognized_phrase)
        if not recognized:
            return {
                "status": "incomplete",
                "expected_phrase": expected_phrase,
                "recognized_phrase": recognized_phrase,
                "exact_match": None,
                "similarity": None,
                "missing_words": [],
                "extra_words": [],
                "phrase_verification_status": "incomplete",
            }

        exact_match = expected == recognized
        similarity = 1.0 if exact_match else 0.0

        expected_tokens = expected.split()
        recognized_tokens = recognized.split()

        missing_words = [word for word in expected_tokens if word not in recognized_tokens]
        extra_words = [word for word in recognized_tokens if word not in expected_tokens]

        status = "completed" if recognized_phrase else "incomplete"
        if recognized_phrase is not None and not exact_match:
            status = "fail"

        return {
            "status": status,
            "expected_phrase": expected_phrase,
            "recognized_phrase": recognized_phrase,
            "exact_match": exact_match,
            "similarity": similarity,
            "missing_words": missing_words,
            "extra_words": extra_words,
            "phrase_verification_status": "pass" if exact_match else "fail",
        }
