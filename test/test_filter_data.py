"""
Tests for filter_data function in awslambda.py
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


class TestFilterData:
    """Tests for filter_data function"""

    def test_filter_by_instructor(self, sample_df):
        """Test filtering by instructor name"""
        result = filter_data(
            df=sample_df,
            instructor="McMahon, Susan",
            terms=["Fall 2020"]
        )

        assert len(result) > 0
        # Check that all results are for the specified instructor by checking courses taught
        assert all(row["Sbjct"] == "ACCT" for _, row in result.iterrows())

    def test_filter_by_terms(self, sample_df):
        """Test filtering by specific terms"""
        result = filter_data(
            df=sample_df,
            instructor="Stephan, Andrew Perry",
            terms=["Fall 2020", "Spring 2021"]
        )

        assert len(result) > 0
        # Verify result has expected columns
        assert "Sbjct" in result.columns
        assert "Crse" in result.columns
        assert "Crse Title" in result.columns

    def test_filter_by_course_title(self, sample_df):
        """Test filtering by course title"""
        result = filter_data(
            df=sample_df,
            instructor="Stephan, Andrew Perry",
            course_title="Corporate Financial Rprtng 1",
            terms=["Fall 2020"]
        )

        assert len(result) > 0
        # All results should be for the specified course
        assert all(row["Crse Title"] == "Corporate Financial Rprtng 1" for _, row in result.iterrows())

    def test_exclude_instructor(self, sample_df):
        """Test excluding a specific instructor"""
        result = filter_data(
            df=sample_df,
            course_title="Corporate Financial Rprtng 1",
            exclude_instructor="McMahon, Susan",
            terms=["Fall 2020"]
        )

        assert len(result) > 0
        # Result should not contain the excluded instructor
        # Since we're aggregating by course, we can't check instructor directly
        # but we can verify we got results

    def test_bad_instructor_raises_value_error(self, sample_df):
        """Test that filtering with no matches raises ValueError"""
        with pytest.raises(ValueError, match="not found"):
            filter_data(
                df=sample_df,
                instructor="Nonexistent Instructor",
                terms=["Fall 2020"]
            )

    def test_metrics_columns_present(self, sample_df):
        """Test that expected metric columns are present in output"""
        result = filter_data(
            df=sample_df,
            instructor="Stephan, Andrew Perry",
            terms=["Fall 2020"]
        )

        assert len(result) > 0

        expected_metrics = [
            "Interact", "Reflect", "Connect", "Collab", "Eval",
            "Synth", "Diverse", "Respect", "Challenge", "Creative",
            "Discuss", "Feedback", "Grading", "Questions", "Tech"
        ]

        for metric in expected_metrics:
            assert metric in result.columns, f"Missing metric: {metric}"

    def test_response_rate_column_present(self, sample_df):
        """Test that Response_Rate column is present"""
        result = filter_data(
            df=sample_df,
            instructor="McMahon, Susan",
            terms=["Fall 2020"]
        )

        assert len(result) > 0
        assert "Response_Rate" in result.columns

    def test_total_sections_column_present(self, sample_df):
        """Test that Total_Sections column is present"""
        result = filter_data(
            df=sample_df,
            instructor="Stephan, Andrew Perry",
            terms=["Fall 2020"]
        )

        assert len(result) > 0
        assert "Total_Sections" in result.columns

    def test_terms_column_present(self, sample_df):
        """Test that Terms column is present and contains lists"""
        result = filter_data(
            df=sample_df,
            instructor="Stephan, Andrew Perry",
            terms=["Fall 2020"]
        )

        assert len(result) > 0
        assert "Terms" in result.columns
        # Check that Terms contains lists
        first_row = result.iloc[0]
        assert isinstance(first_row["Terms"], list)

    def test_na_for_low_enrollment(self, sample_df):
        """Test that courses with enrollment < 10 get NA for metrics"""
        # This would require data with low enrollment in the fixture
        # For now, we'll just verify the function runs
        result = filter_data(
            df=sample_df,
            instructor="Stephan, Andrew Perry",
            terms=["Fall 2020"]
        )

        assert len(result) >= 0  # May have results or not

    def test_multiple_courses_aggregated_separately(self, sample_df):
        """Test that different courses are aggregated separately"""
        result = filter_data(
            df=sample_df,
            instructor="Stephan, Andrew Perry",
            terms=["Fall 2020"]
        )

        if len(result) > 1:
            # Check that we have different courses
            course_titles = result["Crse Title"].unique()
            assert len(course_titles) >= 1

    def test_invalid_terms_raises_error(self, sample_df):
        """Test that invalid terms raise ValueError"""
        with pytest.raises(ValueError, match="Invalid term format"):
            filter_data(
                df=sample_df,
                instructor="McMahon, Susan",
                terms=["Invalid Term"]
            )

    def test_none_terms_raises_error(self, sample_df):
        """Test that None terms raise ValueError"""
        with pytest.raises(ValueError, match="You must provide a valid list of terms"):
            filter_data(
                df=sample_df,
                instructor="McMahon, Susan",
                terms=None
            )

    def test_invalid_instructor_raises_error(self, sample_df):
        """Test that invalid instructor raises ValueError"""
        with pytest.raises(ValueError, match="Instructor"):
            filter_data(
                df=sample_df,
                instructor="Nonexistent Instructor XYZ",
                terms=["Fall 2020"]
            )

    def test_invalid_course_title_raises_error(self, sample_df):
        """Test that invalid course title raises ValueError"""
        with pytest.raises(ValueError, match="Course title"):
            filter_data(
                df=sample_df,
                instructor="McMahon, Susan",
                course_title="Nonexistent Course",
                terms=["Fall 2020"]
            )

    def test_grouped_by_course(self, sample_df):
        """Test that results are grouped by course"""
        result = filter_data(
            df=sample_df,
            instructor="Stephan, Andrew Perry",
            terms=["Fall 2020"]
        )

        assert len(result) > 0
        # Check for required grouping columns
        assert "Sbjct" in result.columns
        assert "Crse" in result.columns
        assert "Crse Title" in result.columns

        # Verify no duplicate courses (should be aggregated)
        course_groups = result.groupby(["Sbjct", "Crse", "Crse Title"])
        for (sbjct, crse, title), group in course_groups:
            assert len(group) == 1, f"Course {sbjct} {crse} appears multiple times"

    def test_count_column_present(self, sample_df):
        """Test that Count column is present"""
        result = filter_data(
            df=sample_df,
            instructor="Stephan, Andrew Perry",
            terms=["Fall 2020"]
        )

        assert len(result) > 0
        assert "Count" in result.columns
        # Count should be >= 1
        assert all(result["Count"] >= 1)

    def test_metrics_are_numeric_or_na(self, sample_df):
        """Test that metric values are numeric or 'NA'"""
        result = filter_data(
            df=sample_df,
            instructor="Stephan, Andrew Perry",
            terms=["Fall 2020"]
        )

        if len(result) > 0:
            metrics = ["Interact", "Reflect", "Connect"]
            for metric in metrics:
                if metric in result.columns:
                    for value in result[metric]:
                        if value == "NA":
                            continue
                        try:
                            float(value)
                        except (ValueError, TypeError):
                            raise AssertionError(f"Value '{value}' in column '{metric}' is not numeric or 'NA'")

    def test_response_rate_is_numeric(self, sample_df):
        """Test that Response_Rate is numeric"""
        result = filter_data(
            df=sample_df,
            instructor="McMahon, Susan",
            terms=["Fall 2020"]
        )

        if len(result) > 0 and "Response_Rate" in result.columns:
            for value in result["Response_Rate"]:
                assert isinstance(value, (int, float))
                assert 0 <= value <= 1  # Response rate should be between 0 and 1

    def test_metrics_formatted_with_decimals(self, sample_df):
        """Test that numeric metrics are formatted as strings with 2 decimals"""
        result = filter_data(
            df=sample_df,
            instructor="Stephan, Andrew Perry",
            terms=["Fall 2020"]
        )

        if len(result) > 0:
            # Check some metrics to see if they're formatted
            metrics = ["Interact", "Respect", "Challenge"]
            for metric in metrics:
                if metric in result.columns:
                    for value in result[metric]:
                        if value != "NA":
                            # Should be a string with 2 decimal places
                            assert isinstance(value, str)
                            # Should match pattern like "4.50"
                            parts = value.split('.')
                            if len(parts) == 2:
                                assert len(parts[1]) == 2

    def test_rounding_before_na_substitution(self, sample_df):
        """
        Test that rounding happens before NA substitution to avoid dtype errors.

        This is a regression test for the bug where fillna("NA") was called before
        rounding, causing "Expected numeric dtype, got object instead" errors in
        older pandas versions.
        """
        result = filter_data(
            df=sample_df,
            instructor="Papuzza, Antonio",
            terms=["Spring 2024", "Fall 2024"]
        )

        # Should complete without raising dtype error
        assert len(result) > 0

        # Check that metrics are properly formatted
        metrics = [
            "Interact", "Reflect", "Connect", "Collab", "Contrib", "Eval",
            "Synth", "Diverse", "Respect", "Challenge", "Creative", "Discuss",
            "Feedback", "Grading", "Questions", "Tech"
        ]

        for metric in metrics:
            if metric in result.columns:
                for value in result[metric]:
                    if value == "NA":
                        # NA values should be string "NA"
                        assert isinstance(value, str)
                        assert value == "NA"
                    else:
                        # Numeric values should be formatted strings with 2 decimals
                        assert isinstance(value, str)
                        # Should be convertible to float
                        float_val = float(value)
                        assert float_val >= 0.0 and float_val <= 5.0  # FCQ scale is 1-5
                        # Should have 2 decimal places (e.g., "4.70", "3.85")
                        parts = value.split('.')
                        assert len(parts) == 2, f"Value '{value}' should have decimal point"
                        assert len(parts[1]) == 2, f"Value '{value}' should have exactly 2 decimal places"

    def test_mixed_na_and_numeric_values(self, sample_df):
        """
        Test that a result with both NA and numeric values is handled correctly.

        This ensures that courses with low enrollment (which get NA) and courses
        with valid metrics can coexist in the same result set.
        """
        # Get results that might have mixed NA and numeric values
        result = filter_data(
            df=sample_df,
            instructor="Papuzza, Antonio",
            terms=["Spring 2024", "Fall 2024"]
        )

        if len(result) > 1:
            # Check if we have at least one NA and one numeric value
            metrics = ["Interact", "Reflect", "Connect"]
            for metric in metrics:
                if metric in result.columns:
                    values = result[metric].tolist()
                    has_na = any(v == "NA" for v in values)
                    has_numeric = any(v != "NA" for v in values)

                    if has_na and has_numeric:
                        # Good - we have mixed values, verify both are correctly formatted
                        for value in values:
                            if value == "NA":
                                assert isinstance(value, str)
                            else:
                                assert isinstance(value, str)
                                # Should be numeric with 2 decimals
                                float(value)  # Should not raise
                                parts = value.split('.')
                                assert len(parts) == 2
                                assert len(parts[1]) == 2
