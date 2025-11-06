"""
Test for histogram data extraction
"""
import sys
import os
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

if __name__ == "__main__":
    test_get_section_scores_for_histogram()
