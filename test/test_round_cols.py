"""
Unit tests for the round_cols function in awslambda.py

These tests specifically verify the fix for the dtype bug where rounding
was attempted on object dtype columns.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import awslambda
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from awslambda import round_cols


class TestRoundCols:
    """Tests for round_cols function"""

    def test_round_numeric_columns(self):
        """Test that numeric columns are properly rounded"""
        df = pd.DataFrame({
            'metric1': [4.7777, 3.8541, 4.123],
            'metric2': [2.999, 5.001, 3.456],
            'other_col': ['A', 'B', 'C']
        })

        result = round_cols(df, round_these_cols=['metric1', 'metric2'], decimals=2)

        # Check that values are rounded and formatted as strings with trailing zeros
        assert result['metric1'].iloc[0] == '4.78'
        assert result['metric1'].iloc[1] == '3.85'
        assert result['metric1'].iloc[2] == '4.12'

        assert result['metric2'].iloc[0] == '3.00'
        assert result['metric2'].iloc[1] == '5.00'
        assert result['metric2'].iloc[2] == '3.46'

        # Other columns should be unchanged
        assert result['other_col'].iloc[0] == 'A'

    def test_round_with_nan_values(self):
        """Test that NaN values are preserved during rounding"""
        df = pd.DataFrame({
            'metric1': [4.777, np.nan, 3.123],
            'metric2': [2.999, 5.001, np.nan]
        })

        result = round_cols(df, round_these_cols=['metric1', 'metric2'], decimals=2)

        # Numeric values should be rounded strings
        assert result['metric1'].iloc[0] == '4.78'
        assert result['metric1'].iloc[2] == '3.12'

        # NaN should still be NaN (not "nan" string)
        assert pd.isna(result['metric1'].iloc[1])
        assert pd.isna(result['metric2'].iloc[2])

    def test_round_preserves_trailing_zeros(self):
        """Test that trailing zeros are preserved (e.g., 4.70 not 4.7)"""
        df = pd.DataFrame({
            'metric': [4.7, 3.0, 2.50]
        })

        result = round_cols(df, round_these_cols=['metric'], decimals=2)

        # All should have exactly 2 decimal places
        assert result['metric'].iloc[0] == '4.70'
        assert result['metric'].iloc[1] == '3.00'
        assert result['metric'].iloc[2] == '2.50'

    def test_round_does_not_modify_object_dtype_with_na(self):
        """
        Test that round_cols works on numeric dtypes with NaN values.

        This is the key test for the bug fix - we should round BEFORE
        converting NaN to "NA" strings.
        """
        # Start with numeric dtype
        df = pd.DataFrame({
            'metric': [4.777, 3.854, np.nan]
        })

        # Verify it starts as numeric
        assert pd.api.types.is_numeric_dtype(df['metric'])

        # Round should work fine on numeric dtype even with NaN
        result = round_cols(df, round_these_cols=['metric'], decimals=2)

        # Check rounded values
        assert result['metric'].iloc[0] == '4.78'
        assert result['metric'].iloc[1] == '3.85'

        # NaN should be preserved as NaN (not converted to string yet)
        assert pd.isna(result['metric'].iloc[2])

    def test_round_with_empty_columns_list(self):
        """Test that empty round_these_cols list doesn't modify dataframe"""
        df = pd.DataFrame({
            'metric1': [4.777, 3.854],
            'metric2': [2.999, 5.001]
        })

        result = round_cols(df, round_these_cols=[], decimals=2)

        # Should be unchanged (still numeric)
        assert pd.api.types.is_numeric_dtype(result['metric1'])
        assert pd.api.types.is_numeric_dtype(result['metric2'])

    def test_round_only_specified_columns(self):
        """Test that only specified columns are rounded"""
        df = pd.DataFrame({
            'metric1': [4.777, 3.854],
            'metric2': [2.999, 5.001],
            'metric3': [1.234, 5.678]
        })

        result = round_cols(df, round_these_cols=['metric1', 'metric2'], decimals=2)

        # metric1 and metric2 should be rounded strings
        assert result['metric1'].iloc[0] == '4.78'
        assert result['metric2'].iloc[0] == '3.00'

        # metric3 should be unchanged (still numeric)
        assert pd.api.types.is_numeric_dtype(result['metric3'])
        assert result['metric3'].iloc[0] == 1.234

    def test_round_different_decimal_places(self):
        """Test rounding with different decimal places"""
        df = pd.DataFrame({
            'metric': [4.12345, 3.98765]
        })

        # Test with 3 decimals
        result = round_cols(df, round_these_cols=['metric'], decimals=3)
        assert result['metric'].iloc[0] == '4.123'
        assert result['metric'].iloc[1] == '3.988'

        # Test with 1 decimal
        df2 = pd.DataFrame({
            'metric': [4.12345, 3.98765]
        })
        result2 = round_cols(df2, round_these_cols=['metric'], decimals=1)
        assert result2['metric'].iloc[0] == '4.1'
        assert result2['metric'].iloc[1] == '4.0'
