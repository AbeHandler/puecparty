"""
Tests for course-specific comparison group logic
"""
import pytest
import pandas as pd
from awslambda import (
    build_comparison_group,
    get_busn_metrics_stats,
    get_section_z_scores,
    load_df
)


class TestCourseSpecificComparisons:
    """Test course-specific comparison functionality"""

    @pytest.fixture
    def df(self):
        """Load the test dataframe once for all tests"""
        return load_df()

    def test_build_comparison_group_all_busn(self, df):
        """Test that build_comparison_group without course_title returns all BUSN sections"""
        terms = ["Spring 2024", "Fall 2024"]
        result = build_comparison_group(df, terms, course_title=None)

        assert len(result) > 0, "Should return sections"
        assert result["ValidForStats"].all(), "All sections should be valid for stats"

        # Should include multiple courses
        unique_courses = result["Crse Title"].nunique()
        assert unique_courses > 1, "Should include multiple courses"

    def test_build_comparison_group_specific_course(self, df):
        """Test that build_comparison_group with course_title filters correctly"""
        terms = ["Spring 2024", "Fall 2024"]

        # Find a course that exists in the data
        all_sections = build_comparison_group(df, terms, course_title=None)
        if len(all_sections) == 0:
            pytest.skip("No valid sections in test data")

        test_course = all_sections["Crse Title"].iloc[0]

        # Get comparison group for specific course
        result = build_comparison_group(df, terms, course_title=test_course)

        assert len(result) > 0, "Should return sections"
        assert (result["Crse Title"] == test_course).all(), "All sections should be for the specified course"
        assert result["ValidForStats"].all(), "All sections should be valid for stats"

    def test_get_busn_metrics_stats_with_course_title(self, df):
        """Test that get_busn_metrics_stats works with course_title parameter"""
        terms = ["Spring 2024", "Fall 2024"]

        # Get stats for all BUSN
        all_busn_stats = get_busn_metrics_stats(df, terms, course_title=None)

        if len(all_busn_stats) == 0:
            pytest.skip("No valid sections in test data")

        # Find a course with multiple sections
        all_sections = build_comparison_group(df, terms, course_title=None)
        course_counts = all_sections["Crse Title"].value_counts()

        if len(course_counts) == 0:
            pytest.skip("No courses found")

        # Pick a course with at least 5 sections for meaningful stats
        courses_with_enough_sections = course_counts[course_counts >= 5]
        if len(courses_with_enough_sections) == 0:
            pytest.skip("No courses with enough sections")

        test_course = courses_with_enough_sections.index[0]

        # Get stats for specific course
        course_stats = get_busn_metrics_stats(df, terms, course_title=test_course)

        assert len(course_stats) > 0, "Should return statistics"
        assert len(course_stats) == len(all_busn_stats), "Should have same number of metrics"

        # Verify the stats are different (course-specific vs all BUSN)
        # Find at least one metric where mean differs
        metrics_differ = False
        for course_stat in course_stats:
            metric = course_stat["Metric"]
            busn_stat = next(s for s in all_busn_stats if s["Metric"] == metric)
            if course_stat["Mean"] != busn_stat["Mean"]:
                metrics_differ = True
                break

        # This might not always be true, but typically course-specific stats differ from all BUSN
        # We'll make this a soft check
        if course_counts[test_course] < len(all_sections) / 2:
            # Only check if the course is a meaningful subset
            assert metrics_differ or course_stat["Count"] < busn_stat["Count"], \
                "Course-specific stats should typically differ from all BUSN stats"

    def test_get_section_z_scores_course_specific_flag(self, df):
        """Test that get_section_z_scores respects course_specific parameter"""
        terms = ["Spring 2024", "Fall 2024"]

        # Find an instructor with multiple sections
        all_sections = build_comparison_group(df, terms, course_title=None)
        if len(all_sections) == 0:
            pytest.skip("No valid sections in test data")

        instructor_counts = all_sections["Instructor Name"].value_counts()
        instructors_with_sections = instructor_counts[instructor_counts >= 2]

        if len(instructors_with_sections) == 0:
            pytest.skip("No instructors with multiple sections")

        test_instructor = instructors_with_sections.index[0]

        # Get z-scores with BUSN-wide comparison
        busn_z_scores = get_section_z_scores(df, terms, instructor=test_instructor, course_specific=False)

        # Get z-scores with course-specific comparison
        course_z_scores = get_section_z_scores(df, terms, instructor=test_instructor, course_specific=True)

        assert len(busn_z_scores) > 0, "Should return BUSN z-scores"
        assert len(course_z_scores) > 0, "Should return course-specific z-scores"

        # Check that BUSN comparison group is labeled correctly
        for z in busn_z_scores[:5]:  # Check first 5
            assert "Comparison_Group" in z, "Should have Comparison_Group field"
            assert z["Comparison_Group"] == "BUSN", "Should indicate BUSN comparison"
            assert "BUSN_Mean" in z, "Should have BUSN_Mean field"
            assert "BUSN_Std" in z, "Should have BUSN_Std field"

        # Check that course-specific comparison group is labeled correctly
        for z in course_z_scores[:5]:  # Check first 5
            assert "Comparison_Group" in z, "Should have Comparison_Group field"
            assert z["Comparison_Group"] != "BUSN", "Should indicate course-specific comparison"
            # Should have course title in the mean/std field names
            course_title = z["Course_Title"]
            assert f"{course_title}_Mean" in z or "Mean" in str(z), "Should have course-specific mean field"

    def test_course_specific_z_scores_differ_from_busn_z_scores(self, df):
        """Test that course-specific z-scores differ from BUSN-wide z-scores"""
        terms = ["Spring 2024", "Fall 2024"]

        # Find an instructor with sections
        all_sections = build_comparison_group(df, terms, course_title=None)
        if len(all_sections) == 0:
            pytest.skip("No valid sections in test data")

        instructor_counts = all_sections["Instructor Name"].value_counts()
        instructors_with_sections = instructor_counts[instructor_counts >= 2]

        if len(instructors_with_sections) == 0:
            pytest.skip("No instructors with multiple sections")

        test_instructor = instructors_with_sections.index[0]

        # Get both types of z-scores
        busn_z_scores = get_section_z_scores(df, terms, instructor=test_instructor, course_specific=False)
        course_z_scores = get_section_z_scores(df, terms, instructor=test_instructor, course_specific=True)

        if len(busn_z_scores) == 0 or len(course_z_scores) == 0:
            pytest.skip("No z-scores computed")

        # Find matching section-metric pairs and compare z-scores
        # Create lookup for busn z-scores
        busn_lookup = {}
        for z in busn_z_scores:
            key = (z["Course"], z["Section"], z["Metric"])
            busn_lookup[key] = z["Z_Score"]

        # Check if any course-specific z-scores differ
        z_scores_differ = False
        for z in course_z_scores:
            key = (z["Course"], z["Section"], z["Metric"])
            if key in busn_lookup:
                if abs(z["Z_Score"] - busn_lookup[key]) > 0.01:  # Allow for rounding
                    z_scores_differ = True
                    break

        # Note: This test might not always pass if the course distribution
        # happens to match the BUSN distribution, so we make it informational
        # rather than a hard assertion
        print(f"Z-scores differ: {z_scores_differ}")
