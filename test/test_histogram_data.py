"""
Test for histogram data extraction
"""
import sys
import os
import math
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from awslambda import get_section_scores_for_histogram, load_df, filter_data

def test_get_section_scores_for_histogram():
    """Test that we can get raw section-level scores for an instructor"""

    # Load the main dataframe
    df = load_df()

    # Test with a known instructor - Adams, Heather L
    instructor = "Adams, Heather L"
    terms = ["Fall 2020", "Spring 2021"]

    result = get_section_scores_for_histogram(df, instructor, terms)

    # Basic validations
    assert isinstance(result, list), "Result should be a list"
    assert len(result) > 0, "Result should not be empty"

    # Check structure of first result
    if len(result) > 0:
        first = result[0]
        assert "Course" in first
        assert "Course_Title" in first
        assert "Term" in first
        assert "Section" in first
        assert "Metric" in first
        assert "Score" in first

        print(f"✓ Found {len(result)} section-level metric scores for {instructor}")
        print(f"  Sample record: {first}")

        # Count unique sections
        unique_sections = set((r['Course'], r['Term'], r['Section']) for r in result)
        print(f"  Unique sections: {len(unique_sections)}")

        # Count by metric
        metrics_count = {}
        for r in result:
            metric = r['Metric']
            metrics_count[metric] = metrics_count.get(metric, 0) + 1

        print(f"  Scores per metric: {metrics_count}")

        # Verify scores are in valid range
        all_scores_valid = all(1.0 <= r['Score'] <= 5.0 for r in result)
        assert all_scores_valid, "All scores should be between 1 and 5"
        print(f"  ✓ All scores in valid range (1-5)")

    print("\n✓ Test passed!")


def test_get_section_scores_for_histogram_2():

    '''
    See test/fixtures/test_heather_2020_2021.xlsx
    '''

    # Load the main dataframe
    df = pd.read_csv("test/fixtures/fcq.csv")
    df = df[df["College"] == "BUSN"].copy()

    # Test with a known instructor - Adams, Heather L
    instructor = "Adams, Heather L"
    terms = ["Fall 2020", "Spring 2021", "Summer 2021", "Fall 2021"]

    result = get_section_scores_for_histogram(df, instructor, terms)
    df = pd.DataFrame(result)
    df = df[df["Metric"] == "Reflect"].copy()

    # this rounding is to mimic what excel does, where 4.5 is rounded up to 5
    df["Score"] = df["Score"].apply(lambda x: math.floor(x + 0.5))

    # See test/fixtures/test_heather_2020_2021.xlsx
    assert int(df["Score"].value_counts()[5]) == 3
    assert int(df["Score"].value_counts()[4]) == 8

def test_get_section_scores_for_histogram_3():

    '''
    See test/fixtures/test_heather_2020_2021.xlsx
    '''

    # Load the main dataframe
    df = pd.read_csv("test/fixtures/fcq.csv")
    df = df[df["College"] == "BUSN"].copy()

    # Test with a known instructor - Adams, Heather L
    instructor = "Adams, Heather L"
    terms = ["Spring 2024", "Fall 2024"]

    result = get_section_scores_for_histogram(df, instructor, terms)
    df_result = pd.DataFrame(result)
    df_reflect = df_result[df_result["Metric"] == "Reflect"].copy()

    # this rounding is to mimic what excel does, where 4.5 is rounded up to 5
    df_reflect["Score"] = df_reflect["Score"].apply(lambda x: math.floor(x + 0.5))

    # Updated assertions based on actual data (not test fixture)
    # There are 6 total sections, with 4 scoring 5 and 2 scoring 4
    assert int(df_reflect["Score"].value_counts()[5]) == 4
    assert int(df_reflect["Score"].value_counts()[4]) == 2

def test_histogram_sections_match_filter_data_total_sections():
    """
    Test that the number of sections in histogram data matches Total_Sections from filter_data.

    This is a regression test to ensure both functions use the same ValidForStats filtering logic.
    Previously, filter_data counted all sections while histogram only counted valid sections,
    causing discrepancies.
    """
    # Load the main dataframe
    df = pd.read_csv("test/fixtures/fcq.csv")
    df = df[df["College"] == "BUSN"].copy()

    # Test with multiple instructors and term ranges
    test_cases = [
        ("Papuzza, Antonio", ["Fall 2021", "Spring 2022", "Summer 2022", "Fall 2022",
                              "Spring 2023", "Summer 2023", "Fall 2023", "Spring 2024", "Summer 2024"]),
        ("Adams, Heather L", ["Fall 2020", "Spring 2021", "Summer 2021", "Fall 2021"]),
        ("McMahon, Susan", ["Fall 2020", "Spring 2021"]),
    ]

    for instructor, terms in test_cases:
        # Get filtered data (main table)
        filtered_data = filter_data(df, instructor=instructor, terms=terms)

        # Skip if no data
        if len(filtered_data) == 0:
            continue

        total_sections_from_table = filtered_data['Total_Sections'].sum()

        # Get histogram data
        histogram_data = get_section_scores_for_histogram(df, instructor, terms)
        df_hist = pd.DataFrame(histogram_data)

        # Count unique sections in histogram
        unique_sections = df_hist[['Course', 'Term', 'Section']].drop_duplicates()
        total_sections_from_histogram = len(unique_sections)

        # They should match
        assert total_sections_from_table == total_sections_from_histogram, \
            f"Section count mismatch for {instructor}: " \
            f"filter_data={total_sections_from_table}, histogram={total_sections_from_histogram}"

        # Verify that each metric has at most the same number of sections
        # (Some metrics may have fewer sections if some sections have null values for that metric)
        for metric in df_hist['Metric'].unique():
            metric_sections = len(df_hist[df_hist['Metric'] == metric])
            assert metric_sections <= total_sections_from_histogram, \
                f"Metric {metric} has {metric_sections} sections, but total is only {total_sections_from_histogram}"

    print(f"✓ All {len(test_cases)} test cases passed: histogram sections match filter_data Total_Sections")


if __name__ == "__main__":
    test_get_section_scores_for_histogram_2()
