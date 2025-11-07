"""
Tests for get_instructor_z_scores function.

This function calculates z-scores (standard deviations from mean) for an instructor's
metrics compared to all BUSN instructors during a given time period.
"""

import pytest
import pandas as pd
import sys
import os

# Add parent directory to path to import awslambda
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from awslambda import get_instructor_z_scores, get_busn_metrics_stats


@pytest.fixture
def sample_df():
    """Load sample FCQ data from CSV fixture"""
    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'fcq.csv')
    df = pd.read_csv(fixture_path)
    # Filter to BUSN only (as done in load_df)
    df = df[df["College"] == "BUSN"].copy()
    return df


class TestInstructorZScores:
    """Tests for instructor z-scores function"""

    def test_z_scores_returns_all_metrics(self, sample_df):
        """Test that z-scores are calculated for all available metrics"""

        terms = ["Fall 2022", "Spring 2023"]
        instructor = "Handler, Abram"

        result = get_instructor_z_scores(sample_df, instructor, terms)

        # Should return metrics (might be less than 16 if instructor doesn't have all)
        assert len(result) > 0
        assert len(result) <= 16

    def test_z_scores_has_required_fields(self, sample_df):
        """Test that each result has all required fields"""

        terms = ["Fall 2022", "Spring 2023"]
        instructor = "Handler, Abram"

        result = get_instructor_z_scores(sample_df, instructor, terms)

        assert len(result) > 0

        for metric_stat in result:
            assert "Metric" in metric_stat
            assert "Instructor_Score" in metric_stat
            assert "BUSN_Mean" in metric_stat
            assert "BUSN_Std" in metric_stat
            assert "Z_Score" in metric_stat
            assert "Instructor_Count" in metric_stat

            # Check types
            assert isinstance(metric_stat["Metric"], str)
            assert isinstance(metric_stat["Instructor_Score"], float)
            assert isinstance(metric_stat["BUSN_Mean"], float)
            assert isinstance(metric_stat["BUSN_Std"], float)
            assert isinstance(metric_stat["Z_Score"], float)
            assert isinstance(metric_stat["Instructor_Count"], int)

    def test_z_score_calculation_is_correct(self, sample_df):
        """Test that z-score calculation is mathematically correct"""

        terms = ["Fall 2022", "Spring 2023"]
        instructor = "Handler, Abram"

        result = get_instructor_z_scores(sample_df, instructor, terms)

        for metric_stat in result:
            # Manually calculate z-score
            expected_z = (metric_stat["Instructor_Score"] - metric_stat["BUSN_Mean"]) / metric_stat["BUSN_Std"]
            expected_z = round(expected_z, 2)

            # Compare with returned z-score
            assert abs(metric_stat["Z_Score"] - expected_z) < 0.01, \
                f"Z-score mismatch for {metric_stat['Metric']}: expected {expected_z}, got {metric_stat['Z_Score']}"

    def test_z_score_positive_when_above_mean(self, sample_df):
        """Test that z-scores are positive when instructor is above BUSN mean"""

        terms = ["Fall 2022", "Spring 2023", "Fall 2023"]
        instructor = "Handler, Abram"

        result = get_instructor_z_scores(sample_df, instructor, terms)

        # Find metrics where instructor is above mean
        above_mean = [m for m in result if m["Instructor_Score"] > m["BUSN_Mean"]]

        for metric_stat in above_mean:
            assert metric_stat["Z_Score"] > 0, \
                f"{metric_stat['Metric']}: score {metric_stat['Instructor_Score']} > mean {metric_stat['BUSN_Mean']} should have positive z-score"

    def test_z_score_negative_when_below_mean(self, sample_df):
        """Test that z-scores are negative when instructor is below BUSN mean"""

        terms = ["Fall 2022", "Spring 2023", "Fall 2023"]
        instructor = "Handler, Abram"

        result = get_instructor_z_scores(sample_df, instructor, terms)

        # Find metrics where instructor is below mean
        below_mean = [m for m in result if m["Instructor_Score"] < m["BUSN_Mean"]]

        for metric_stat in below_mean:
            assert metric_stat["Z_Score"] < 0, \
                f"{metric_stat['Metric']}: score {metric_stat['Instructor_Score']} < mean {metric_stat['BUSN_Mean']} should have negative z-score"

    def test_z_scores_nonexistent_instructor_returns_empty(self, sample_df):
        """Test that z-scores for non-existent instructor returns empty list"""

        terms = ["Fall 2022", "Spring 2023"]
        instructor = "Nonexistent, Instructor"

        result = get_instructor_z_scores(sample_df, instructor, terms)

        assert result == []

    def test_z_scores_empty_terms_returns_empty(self, sample_df):
        """Test that z-scores for empty time period returns empty list"""

        terms = ["Fall 2015", "Spring 2016"]
        instructor = "Handler, Abram"

        result = get_instructor_z_scores(sample_df, instructor, terms)

        assert result == []

    def test_z_scores_respect_valid_for_stats(self, sample_df):
        """Test that z-scores only use valid sections (enrollment >= 10 for undergrad)"""

        terms = ["Fall 2022", "Spring 2023"]
        instructor = "Handler, Abram"

        result = get_instructor_z_scores(sample_df, instructor, terms)

        # Verify counts match ValidForStats logic
        df_filtered = sample_df.copy()
        df_filtered["Year"] = df_filtered["Year"].astype(int)
        df_filtered["Term_Year"] = df_filtered["Term"] + " " + df_filtered["Year"].astype(str)
        df_filtered = df_filtered[df_filtered["Instructor Name"] == instructor]
        df_filtered = df_filtered[df_filtered["Term_Year"].isin(terms)]

        df_filtered["Crse"] = df_filtered["Crse"].astype(int)
        first_digit = df_filtered["Crse"].astype(str).str[0].astype(int)
        df_filtered["ValidForStats"] = ((df_filtered["Enroll"] >= 10) & (first_digit <= 4)) | (first_digit > 4)

        valid_sections = df_filtered[df_filtered["ValidForStats"]]

        # Check for a specific metric
        interact_count = valid_sections["Interact"].dropna().count()
        interact_result = next((r for r in result if r["Metric"] == "Interact"), None)

        if interact_result:
            assert interact_result["Instructor_Count"] == interact_count

    def test_z_scores_values_rounded_to_2_decimals(self, sample_df):
        """Test that all numeric values are rounded to 2 decimal places"""

        terms = ["Fall 2022", "Spring 2023"]
        instructor = "Handler, Abram"

        result = get_instructor_z_scores(sample_df, instructor, terms)

        for metric_stat in result:
            # Check that values have at most 2 decimal places
            for key in ["Instructor_Score", "BUSN_Mean", "BUSN_Std", "Z_Score"]:
                value_str = str(metric_stat[key])
                if "." in value_str:
                    assert len(value_str.split(".")[1]) <= 2, \
                        f"{key} has more than 2 decimal places: {value_str}"
