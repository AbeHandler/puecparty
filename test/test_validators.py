"""
Tests for validation functions in awslambda.py
"""

import pytest
import sys
import os

# Add parent directory to path to import awslambda
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from awslambda import validiate_mode, validate_terms


class TestValidiateMode:
    """Tests for validiate_mode function"""

    def test_valid_calendar_mode(self):
        """Test that 'calendar' mode is valid"""
        assert validiate_mode("calendar") == True

    def test_valid_academic_mode(self):
        """Test that 'academic' mode is valid"""
        assert validiate_mode("academic") == True

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError"""
        with pytest.raises(ValueError, match="Invalid mode: 'invalid'. Must be 'calendar' or 'academic'"):
            validiate_mode("invalid")

    def test_empty_string_mode_raises_error(self):
        """Test that empty string raises ValueError"""
        with pytest.raises(ValueError, match="Invalid mode: ''. Must be 'calendar' or 'academic'"):
            validiate_mode("")

    def test_case_sensitive_calendar(self):
        """Test that mode is case sensitive - 'Calendar' should fail"""
        with pytest.raises(ValueError, match="Invalid mode: 'Calendar'. Must be 'calendar' or 'academic'"):
            validiate_mode("Calendar")

    def test_case_sensitive_academic(self):
        """Test that mode is case sensitive - 'Academic' should fail"""
        with pytest.raises(ValueError, match="Invalid mode: 'Academic'. Must be 'calendar' or 'academic'"):
            validiate_mode("Academic")

    def test_numeric_mode_raises_error(self):
        """Test that numeric input raises ValueError"""
        with pytest.raises(ValueError):
            validiate_mode(123)

    def test_numeric_mode_raises_error(self):
        """Test that numeric input raises ValueError"""
        with pytest.raises(ValueError):
            validiate_mode(None)

class TestValidateTerms:
    """Tests for validate_terms function"""

    def test_valid_single_term(self):
        """Test validation of a single valid term"""
        result = validate_terms(["Fall 2024"])
        assert result == [("Fall", 2024)]

    def test_valid_multiple_terms(self):
        """Test validation of multiple valid terms"""
        result = validate_terms(["Spring 2024", "Fall 2024"])
        assert result == [("Spring", 2024), ("Fall", 2024)]

    def test_all_valid_term_types(self):
        """Test all three valid term types"""
        result = validate_terms(["Fall 2024", "Spring 2024", "Summer 2024"])
        assert result == [("Fall", 2024), ("Spring", 2024), ("Summer", 2024)]

    def test_none_terms_raises_error(self):
        """Test that None raises ValueError"""
        with pytest.raises(ValueError, match="You must provide a valid list of terms to use the lambda"):
            validate_terms(None)

    def test_invalid_format_no_space(self):
        """Test that term without space raises ValueError"""
        with pytest.raises(ValueError, match="Invalid term format: 'Fall2024'. Expected format: 'Fall 2024'"):
            validate_terms(["Fall2024"])

    def test_invalid_format_wrong_case(self):
        """Test that lowercase term name raises ValueError"""
        with pytest.raises(ValueError, match="Invalid term format: 'fall 2024'. Expected format: 'Fall 2024'"):
            validate_terms(["fall 2024"])

    def test_invalid_term_name(self):
        """Test that invalid term name raises ValueError"""
        with pytest.raises(ValueError, match="Invalid term format: 'Winter 2024'. Expected format: 'Fall 2024'"):
            validate_terms(["Winter 2024"])

    def test_invalid_year_format(self):
        """Test that non-numeric year raises ValueError"""
        with pytest.raises(ValueError, match="Invalid term format: 'Fall ABCD'. Expected format: 'Fall 2024'"):
            validate_terms(["Fall ABCD"])

    def test_year_too_early(self):
        """Test that year before 1900 raises ValueError"""
        with pytest.raises(ValueError, match="Invalid year: 1899"):
            validate_terms(["Fall 1899"])

    def test_year_too_late(self):
        """Test that year after 2030 raises ValueError"""
        with pytest.raises(ValueError, match="Invalid year: 2031"):
            validate_terms(["Fall 2031"])

    def test_year_boundary_1900(self):
        """Test that year 1900 is valid (boundary)"""
        result = validate_terms(["Fall 1900"])
        assert result == [("Fall", 1900)]

    def test_year_boundary_2030(self):
        """Test that year 2030 is valid (boundary)"""
        result = validate_terms(["Fall 2030"])
        assert result == [("Fall", 2030)]

    def test_empty_list(self):
        """Test that empty list returns empty result"""
        result = validate_terms([])
        assert result == []

    def test_mixed_valid_invalid_terms(self):
        """Test that one invalid term causes entire validation to fail"""
        with pytest.raises(ValueError, match="Invalid term format: 'Invalid 2024'"):
            validate_terms(["Fall 2024", "Invalid 2024"])

    def test_extra_spaces(self):
        """Test that extra spaces cause validation to fail"""
        with pytest.raises(ValueError, match="Invalid term format: 'Fall  2024'. Expected format: 'Fall 2024'"):
            validate_terms(["Fall  2024"])

    def test_leading_trailing_spaces(self):
        """Test that leading/trailing spaces cause validation to fail"""
        with pytest.raises(ValueError, match="Invalid term format: ' Fall 2024'. Expected format: 'Fall 2024'"):
            validate_terms([" Fall 2024"])

    def test_two_digit_year(self):
        """Test that two-digit year raises ValueError"""
        with pytest.raises(ValueError, match="Invalid term format: 'Fall 24'. Expected format: 'Fall 2024'"):
            validate_terms(["Fall 24"])

    def test_three_digit_year(self):
        """Test that three-digit year raises ValueError"""
        with pytest.raises(ValueError, match="Invalid term format: 'Fall 202'. Expected format: 'Fall 2024'"):
            validate_terms(["Fall 202"])

    def test_five_digit_year(self):
        """Test that five-digit year raises ValueError"""
        with pytest.raises(ValueError, match="Invalid term format: 'Fall 20241'. Expected format: 'Fall 2024'"):
            validate_terms(["Fall 20241"])
