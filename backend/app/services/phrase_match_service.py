from __future__ import annotations

import difflib
import re
from typing import Any

PHRASE_MATCH_THRESHOLD = 0.90  # Require 90% similarity to pass phrase challenge


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
                "similarity": 0.0,
                "similarity_percentage": 0.0,
                "missing_words": expected.split() if expected else [],
                "extra_words": [],
                "phrase_verification_status": "incomplete",
                "threshold": PHRASE_MATCH_THRESHOLD,
            }

        exact_match = expected == recognized
        expected_tokens = expected.split()
        recognized_tokens = recognized.split()

        # Word-level token overlap
        matched_tokens_count = sum(1 for tok in recognized_tokens if tok in expected_tokens)
        max_tokens = max(len(expected_tokens), len(recognized_tokens), 1)
        token_ratio = matched_tokens_count / max_tokens

        # Character-level sequence ratio
        seq_ratio = difflib.SequenceMatcher(None, expected, recognized).ratio()

        # Final similarity metric (highest of token ratio and character sequence ratio)
        similarity = round(max(seq_ratio, token_ratio), 4)
        similarity_pct = round(similarity * 100.0, 1)

        # 90% similarity threshold check
        passed_90_percent = similarity >= PHRASE_MATCH_THRESHOLD

        missing_words = [word for word in expected_tokens if word not in recognized_tokens]
        extra_words = [word for word in recognized_tokens if word not in expected_tokens]

        status = "completed"
        phrase_verification_status = "pass" if passed_90_percent else "fail"

        return {
            "status": status,
            "expected_phrase": expected_phrase,
            "recognized_phrase": recognized_phrase,
            "exact_match": exact_match,
            "passed_threshold": passed_90_percent,
            "similarity": similarity,
            "similarity_percentage": similarity_pct,
            "threshold": PHRASE_MATCH_THRESHOLD,
            "threshold_percentage": 90.0,
            "missing_words": missing_words,
            "extra_words": extra_words,
            "phrase_verification_status": phrase_verification_status,
        }

