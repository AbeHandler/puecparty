"""
Tests for specific queries run by hand. 

To write a test for this file define your own query such as classes 
for Handler, Abram from 2023 and check that the results match

This is the main file for testing the input/output carefully
"""

import pytest
import pandas as pd
import sys
import os

# Add parent directory to path to import awslambda
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from awslambda import filter_data


@pytest.fixture
def sample_df():
    """Load sample FCQ data from CSV fixture"""
    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'fcq.csv')
    df = pd.read_csv(fixture_path)
    # Filter to BUSN only (as done in load_df)
    df = df[df["College"] == "BUSN"].copy()
    return df


class TestQueries:
    """Tests for specific query scenarios"""

    def test_handler_abram_2022_to_2025(self, sample_df):
        """Test filtering Handler, Abram from 2022 to present returns 6 courses"""

        all_years = sample_df['Year'].unique()
        terms_2022_plus = []
        for year in all_years:
            if year >= 2022 and year < 2026:
                for term in ['Spring', 'Summer', 'Fall']:
                    term_str = f"{term} {year}"
                    # Check if term exists in data
                    if len(sample_df[(sample_df['Term'] == term) & (sample_df['Year'] == year)]) > 0:
                        terms_2022_plus.append(term_str)

        # Handler taught 3 unique courses since 2022 (across diff sections)
        result = filter_data(
            df=sample_df,
            instructor="Handler, Abram",
            terms=terms_2022_plus
        )

        assert len(result) == 3


    def test_handler_abram_2022_to_2025_3220_sections(self, sample_df):
        """Test filtering Handler, Abram from 2022 to present returns 6 courses"""

        all_years = sample_df['Year'].unique()
        terms_2022_plus = []
        for year in all_years:
            if year >= 2022 and year < 2026:
                for term in ['Spring', 'Summer', 'Fall']:
                    term_str = f"{term} {year}"
                    # Check if term exists in data
                    if len(sample_df[(sample_df['Term'] == term) & (sample_df['Year'] == year)]) > 0:
                        terms_2022_plus.append(term_str)

        # Handler taught 3 unique courses since 2022 (across diff sections)
        result = filter_data(
            df=sample_df,
            instructor="Handler, Abram",
            terms=terms_2022_plus
        )

        result = result[["Total_Sections", "Crse Title"]]
        result = result[result["Crse Title"] == "Intro to Python Programming"].copy()
        assert result.iloc[0]["Total_Sections"] == 4, "Handler has taught 4 total sections of 3220, 2022-2025"


    def test_exclude_instructor_without_course_raises_error(self, sample_df):
        """Test that using exclude_instructor without course_title raises ValueError"""

        all_years = sample_df['Year'].unique()
        terms_2022 = []
        for year in all_years:
            if year >= 2022 and year < 2023:
                for term in ['Spring', 'Summer', 'Fall']:
                    term_str = f"{term} {year}"
                    # Check if term exists in data
                    if len(sample_df[(sample_df['Term'] == term) & (sample_df['Year'] == year)]) > 0:
                        terms_2022.append(term_str)

        # Should raise error when exclude_instructor is used without course_title
        with pytest.raises(ValueError, match="You must specify a course to exclude an instructor"):
            filter_data(
                df=sample_df,
                exclude_instructor="Handler, Abram",
                terms=terms_2022
            )

    def test_exclude_instructor_2022_intro_python_returns_one_course(self, sample_df):
        """Test that using exclude_instructor without course_title raises ValueError"""

        all_years = sample_df['Year'].unique()
        terms_2022 = []
        for year in all_years:
            if year >= 2022 and year < 2023:
                for term in ['Spring', 'Summer', 'Fall']:
                    term_str = f"{term} {year}"
                    # Check if term exists in data
                    if len(sample_df[(sample_df['Term'] == term) & (sample_df['Year'] == year)]) > 0:
                        terms_2022.append(term_str)

        df = filter_data(
            df=sample_df,
            exclude_instructor="Handler, Abram",
            course_title="Intro to Python Programming",
            terms=terms_2022
        )
        assert len(df) == 1, "Only 1 course"

    def test_exclude_instructor_2022_intro_python_returns_correct_average(self, sample_df):
        """Test that using exclude_instructor without course_title raises ValueError"""

        all_years = sample_df['Year'].unique()
        terms_2022 = []
        for year in all_years:
            if year >= 2022 and year < 2023:
                for term in ['Spring', 'Summer', 'Fall']:
                    term_str = f"{term} {year}"
                    # Check if term exists in data
                    if len(sample_df[(sample_df['Term'] == term) & (sample_df['Year'] == year)]) > 0:
                        terms_2022.append(term_str)

        df = filter_data(
            df=sample_df,
            exclude_instructor="Handler, Abram",
            course_title="Intro to Python Programming",
            terms=terms_2022
        )
        assert float(df.iloc[0]["Tech"]) == 4.64