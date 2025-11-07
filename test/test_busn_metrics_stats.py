"""
Tests for get_busn_metrics_stats function.

This function calculates mean and standard deviation for each metric
across all BUSN instructors during a given time period.
"""

import pytest
import pandas as pd
import sys
import os

# Add parent directory to path to import awslambda
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from awslambda import get_busn_metrics_stats


@pytest.fixture
def sample_df():
    """Load sample FCQ data from CSV fixture"""
    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'fcq.csv')
    df = pd.read_csv(fixture_path)
    # Filter to BUSN only (as done in load_df)
    df = df[df["College"] == "BUSN"].copy()
    return df


class TestBusnMetricsStats:
    """Tests for BUSN metrics statistics function"""

    def test_busn_metrics_stats_2022_returns_all_metrics(self, sample_df):
        """Test that stats for 2022 returns all 16 metrics"""

        # Build terms list for all of 2022
        all_years = sample_df['Year'].unique()
        terms_2022 = []
        for year in all_years:
            if year >= 2022 and year < 2023:
                for term in ['Spring', 'Summer', 'Fall']:
                    term_str = f"{term} {year}"
                    # Check if term exists in data
                    if len(sample_df[(sample_df['Term'] == term) & (sample_df['Year'] == year)]) > 0:
                        terms_2022.append(term_str)

        result = get_busn_metrics_stats(sample_df, terms_2022)

        # Should return 16 metrics
        assert len(result) == 16

        # Check that all expected metrics are present
        expected_metrics = [
            "Interact", "Reflect", "Connect", "Collab", "Contrib", "Eval",
            "Synth", "Diverse", "Respect", "Challenge", "Creative", "Discuss",
            "Feedback", "Grading", "Questions", "Tech"
        ]
        result_metrics = [r["Metric"] for r in result]
        assert set(result_metrics) == set(expected_metrics)

    def test_busn_metrics_stats_has_required_fields(self, sample_df):
        """Test that each result has Metric, Mean, Std, and Count fields"""

        terms = ["Fall 2022", "Spring 2023"]
        result = get_busn_metrics_stats(sample_df, terms)

        assert len(result) > 0

        for metric_stat in result:
            assert "Metric" in metric_stat
            assert "Mean" in metric_stat
            assert "Std" in metric_stat
            assert "Count" in metric_stat

            # Check types
            assert isinstance(metric_stat["Metric"], str)
            assert isinstance(metric_stat["Mean"], float)
            assert isinstance(metric_stat["Std"], float)
            assert isinstance(metric_stat["Count"], int)

    def test_busn_metrics_stats_values_in_valid_range(self, sample_df):
        """Test that mean and std values are in valid range (1-6 for FCQ scores)"""

        terms = ["Fall 2022", "Spring 2023", "Fall 2023"]
        result = get_busn_metrics_stats(sample_df, terms)

        for metric_stat in result:
            # Mean should be between 1 and 6
            assert 1 <= metric_stat["Mean"] <= 6

            # Std should be non-negative and reasonable
            assert 0 <= metric_stat["Std"] <= 3

            # Count should be positive
            assert metric_stat["Count"] > 0

    def test_busn_metrics_stats_empty_terms_returns_empty(self, sample_df):
        """Test that stats for non-existent terms returns empty list"""

        # Use terms that don't exist in the data (but within valid year range)
        terms = ["Fall 2015", "Spring 2016"]
        result = get_busn_metrics_stats(sample_df, terms)

        assert result == []

    def test_busn_metrics_stats_mean_is_rounded_to_2_decimals(self, sample_df):
        """Test that mean and std are rounded to 2 decimal places"""

        terms = ["Fall 2022"]
        result = get_busn_metrics_stats(sample_df, terms)

        for metric_stat in result:
            # Check that values have at most 2 decimal places
            mean_str = str(metric_stat["Mean"])
            std_str = str(metric_stat["Std"])

            if "." in mean_str:
                assert len(mean_str.split(".")[1]) <= 2
            if "." in std_str:
                assert len(std_str.split(".")[1]) <= 2

    def test_busn_metrics_stats_respects_valid_for_stats(self, sample_df):
        """Test that only valid sections (enrollment >= 10 for undergrad) are included"""

        terms = ["Fall 2022", "Spring 2023"]

        # Get the stats
        result = get_busn_metrics_stats(sample_df, terms)

        # Manually count valid sections for a specific term
        df_filtered = sample_df.copy()
        df_filtered["Year"] = df_filtered["Year"].astype(int)
        df_filtered["Term_Year"] = df_filtered["Term"] + " " + df_filtered["Year"].astype(str)
        df_filtered = df_filtered[df_filtered["Term_Year"].isin(terms)].copy()

        # Apply the same ValidForStats logic
        df_filtered["Crse"] = df_filtered["Crse"].astype(int)
        first_digit = df_filtered["Crse"].astype(str).str[0].astype(int)
        df_filtered["ValidForStats"] = ((df_filtered["Enroll"] >= 10) & (first_digit <= 4)) | (first_digit > 4)

        valid_sections = df_filtered[df_filtered["ValidForStats"]]

        # Count for a specific metric (e.g., "Interact")
        interact_count = valid_sections["Interact"].dropna().count()

        # Find Interact in results
        interact_result = next((r for r in result if r["Metric"] == "Interact"), None)

        assert interact_result is not None
        assert interact_result["Count"] == interact_count
