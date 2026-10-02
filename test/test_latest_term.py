"""Tests for get_latest_term, which drives the "Data current as of" header line."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from awslambda import get_latest_term


def test_fall_beats_spring_in_same_year():
    df = pd.DataFrame({"Year": [2025, 2025, 2024], "Term": ["Spring", "Fall", "Fall"]})
    assert get_latest_term(df) == "Fall 2025"


def test_later_year_beats_earlier_fall():
    df = pd.DataFrame({"Year": [2025, 2026], "Term": ["Fall", "Spring"]})
    assert get_latest_term(df) == "Spring 2026"


def test_summer_between_spring_and_fall():
    df = pd.DataFrame({"Year": [2026, 2026], "Term": ["Spring", "Summer"]})
    assert get_latest_term(df) == "Summer 2026"


def test_fixture_data():
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "fcq.csv")
    df = pd.read_csv(fixture_path)
    assert get_latest_term(df) == "Spring 2025"
