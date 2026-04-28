import pytest
import pandas as pd
import numpy as np
from session_analysis import safety_grade, compute_summary, count_episodes, format_duration

# --- safety_grade ---
def test_safety_grade_excellent():
    assert safety_grade(10) == ("A", "Excellent")

def test_safety_grade_critical():
    assert safety_grade(75) == ("F", "Critical")

def test_safety_grade_boundary():
    assert safety_grade(30)[0] == "C"

# --- format_duration ---
def test_format_duration_seconds():
    assert format_duration(45) == "45s"

def test_format_duration_minutes():
    assert format_duration(125) == "2m 5s"

def test_format_duration_hours():
    assert format_duration(3661) == "1h 1m 1s"

# --- count_episodes ---
def test_count_episodes_basic():
    df = pd.DataFrame({"State": ["ACTIVE", "DROWSY", "DROWSY", "ACTIVE", "DROWSY"]})
    result = count_episodes(df, "DROWSY")
    assert result["count"] == 2
    assert result["total_sec"] == 3
    assert result["max_sec"] == 2

def test_count_episodes_none():
    df = pd.DataFrame({"State": ["ACTIVE", "ACTIVE", "ACTIVE"]})
    result = count_episodes(df, "DROWSY")
    assert result["count"] == 0
    assert result["max_sec"] == 0
