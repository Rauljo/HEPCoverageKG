"""
HEPCoverageKG aliases: Phase B - Semantic Guards

Deterministic vetoes to catch semantic mismatches that vector embeddings miss.
Pure generic functions for string comparison.
"""
from __future__ import annotations

import re

# Regex to find numbers, including versions with dots (e.g. 13, 2.2.1, 5.02)
_NUM_PATTERN = re.compile(r'\d+(?:\.\d+)*')

# Set of standard physics units we care about preventing mismatches on
_UNITS = {"tev", "gev", "mev", "pb", "fb", "nb", "ab"}


def _extract_numbers(s: str) -> set[str]:
    """Return all numerical sequences found in the string."""
    return set(_NUM_PATTERN.findall(s))


def _extract_units(s: str) -> set[str]:
    """Return all known physics units found in the string."""
    tokens = re.split(r"[._\-\s]+", s.lower())
    return set(tokens).intersection(_UNITS)


def passes_semantic_guards(str_a: str, str_b: str) -> tuple[bool, str]:
    """
    Apply hard deterministic rules to veto bad matches.
    Returns (True, "passed") if safe, or (False, "reason") if vetoed.
    """
    # 1. Number Guard
    nums_a = _extract_numbers(str_a)
    nums_b = _extract_numbers(str_b)
    
    # If both strings contain numbers, but their set of numbers differ, it's a mismatch.
    # (e.g., '13 TeV' vs '14 TeV' -> veto. 'MadGraph' vs 'MG5' -> safe, because one has no numbers).
    if nums_a and nums_b and nums_a != nums_b:
        return False, f"number_mismatch ({nums_a} != {nums_b})"
        
    # 2. Unit Guard
    units_a = _extract_units(str_a)
    units_b = _extract_units(str_b)
    
    # If both contain specific physics units, and they differ, it's a mismatch.
    if units_a and units_b and units_a != units_b:
        return False, f"unit_mismatch ({units_a} != {units_b})"
        
    return True, "passed"
