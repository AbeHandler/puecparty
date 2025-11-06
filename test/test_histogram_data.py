"""
Test for histogram data extraction
"""
import sys
import os
import math
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from awslambda import get_section_scores_for_histogram, load_df

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
    return True


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
    return True

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
    df = pd.DataFrame(result)
    df = df[df["Metric"] == "Reflect"].copy()

    # this rounding is to mimic what excel does, where 4.5 is rounded up to 5
    df["Score"] = df["Score"].apply(lambda x: math.floor(x + 0.5))

    # See test/fixtures/test_heather_2020_2021.xlsx
    assert int(df["Score"].value_counts()[5]) == 3
    assert int(df["Score"].value_counts()[4]) == 8
    return True


if __name__ == "__main__":
    test_get_section_scores_for_histogram_2()
