"""
Tests for JSON serialization and NaN handling in awslambda.py

These tests verify that the prepare_for_json function properly handles
NaN values to prevent invalid JSON serialization errors.
"""

import pytest
import pandas as pd
import numpy as np
import json
import sys
import os

# Add parent directory to path to import awslambda
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from awslambda import prepare_for_json, NAN_VALUE, format_data, filter_data


class TestPrepareForJson:
    """Tests for the prepare_for_json function"""

    def test_replaces_numpy_nan_with_none(self):
        """Test that numpy NaN values are replaced with None"""
        df = pd.DataFrame({
            'col1': [1.0, np.nan, 3.0],
            'col2': [np.nan, 5.0, 6.0]
        })

        result = prepare_for_json(df)

        # Check that NaN values are replaced with None
        assert result.loc[1, 'col1'] is None
        assert result.loc[0, 'col2'] is None

    def test_replaces_pd_na_with_none(self):
        """Test that pandas NA values are replaced with None"""
        df = pd.DataFrame({
            'col1': [1.0, pd.NA, 3.0],
            'col2': [pd.NA, 5.0, 6.0]
        })

        result = prepare_for_json(df)

        # Check that NA values are replaced with None
        assert result.loc[1, 'col1'] is None
        assert result.loc[0, 'col2'] is None

    def test_preserves_valid_values(self):
        """Test that valid numeric and string values are preserved"""
        df = pd.DataFrame({
            'numeric': [1.5, 2.5, 3.5],
            'string': ['a', 'b', 'c'],
            'integer': [1, 2, 3]
        })

        result = prepare_for_json(df)

        # All values should be unchanged
        assert result['numeric'].tolist() == [1.5, 2.5, 3.5]
        assert result['string'].tolist() == ['a', 'b', 'c']
        assert result['integer'].tolist() == [1, 2, 3]

    def test_preserves_na_string(self):
        """Test that the string 'NA' is preserved (not replaced with None)"""
        df = pd.DataFrame({
            'col1': ['NA', 'value', 'NA']
        })

        result = prepare_for_json(df)

        # String 'NA' should remain as is
        assert result['col1'].tolist() == ['NA', 'value', 'NA']

    def test_json_serializable_output(self):
        """Test that the output can be serialized to JSON"""
        df = pd.DataFrame({
            'col1': [1.0, np.nan, 3.0],
            'col2': ['a', 'b', NAN_VALUE],
            'col3': [np.nan, pd.NA, None]
        })

        result = prepare_for_json(df)
        records = result.to_dict('records')

        # Should not raise an error
        json_str = json.dumps(records)

        # Verify JSON is valid
        parsed = json.loads(json_str)
        assert len(parsed) == 3

        # Check that NaN became null in JSON
        assert parsed[0]['col1'] == 1.0
        assert parsed[1]['col1'] is None  # was np.nan
        assert parsed[0]['col2'] == 'a'
        assert parsed[2]['col2'] == NAN_VALUE  # string "NA" preserved
        assert parsed[0]['col3'] is None  # was np.nan

    def test_handles_mixed_types(self):
        """Test handling of DataFrame with mixed data types"""
        df = pd.DataFrame({
            'float': [1.5, np.nan, 3.5],
            'int': [1, 2, 3],
            'str': ['a', NAN_VALUE, 'c'],
            'obj': [{'key': 'val'}, None, {'key': 'val2'}]
        })

        result = prepare_for_json(df)
        records = result.to_dict('records')

        # Should serialize without error
        json_str = json.dumps(records)
        assert json_str is not None

    def test_does_not_modify_original_dataframe(self):
        """Test that prepare_for_json doesn't modify the original DataFrame"""
        df = pd.DataFrame({
            'col1': [1.0, np.nan, 3.0]
        })

        # Keep reference to original
        original_value = df.loc[1, 'col1']

        # Call prepare_for_json
        result = prepare_for_json(df)

        # Original should be unchanged
        assert pd.isna(original_value)
        assert pd.isna(df.loc[1, 'col1'])

        # Result should have None
        assert result.loc[1, 'col1'] is None


class TestJsonSerializationIntegration:
    """Integration tests for JSON serialization with real data flow"""

    @pytest.fixture
    def sample_df(self):
        """Load sample FCQ data from CSV fixture"""
        fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'fcq.csv')
        df = pd.read_csv(fixture_path)
        # Filter to BUSN only (as done in load_df)
        df = df[df["College"] == "BUSN"].copy()
        return df

    def test_filter_data_produces_json_serializable_output(self, sample_df):
        """Test that filter_data output can be serialized to JSON after prepare_for_json"""
        # Get filtered data
        filtered_data = filter_data(
            df=sample_df,
            instructor="Papuzza, Antonio",
            terms=["Spring 2024", "Fall 2024"]
        )

        if len(filtered_data) == 0:
            pytest.skip("No data for test instructor")

        # Format the data
        formatted_data = format_data(filtered_data)

        # Prepare for JSON
        prepared_data = prepare_for_json(formatted_data)

        # Convert to records
        records = prepared_data.to_dict('records')

        # Should serialize without error
        json_str = json.dumps(records)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) > 0

        # Verify no "NaN" strings in JSON output
        assert 'NaN' not in json_str
        assert '"Value": NaN' not in json_str

    def test_response_rate_nan_handling(self, sample_df):
        """Test that NaN in Response_Rate column is properly handled"""
        # Create a DataFrame with NaN in Response_Rate
        test_df = sample_df.copy()

        # Get data that might have NaN in Response_Rate
        filtered_data = filter_data(
            df=test_df,
            instructor="Papuzza, Antonio",
            terms=["Spring 2024", "Fall 2024"]
        )

        if len(filtered_data) == 0:
            pytest.skip("No data for test instructor")

        # Manually introduce NaN in Response_Rate before formatting
        filtered_data.loc[filtered_data.index[0], 'Response_Rate'] = np.nan

        # Format converts to long format, so Response_Rate becomes a row
        formatted_data = format_data(filtered_data)

        # This should have NaN in the Value column where Metric is Response_Rate
        prepared_data = prepare_for_json(formatted_data)
        records = prepared_data.to_dict('records')

        # Should serialize without error
        json_str = json.dumps(records)

        # Parse back
        parsed = json.loads(json_str)

        # Find the record where Metric is Response_Rate and Value is None
        found_null = False
        for record in parsed:
            if record.get('Metric') == 'Response_Rate' and record.get('Value') is None:
                found_null = True
                break

        assert found_null, "Should have found at least one null Value for Response_Rate metric"

    def test_total_sections_nan_handling(self, sample_df):
        """Test that NaN in Total_Sections column is properly handled"""
        test_df = sample_df.copy()

        filtered_data = filter_data(
            df=test_df,
            instructor="Papuzza, Antonio",
            terms=["Spring 2024", "Fall 2024"]
        )

        if len(filtered_data) == 0:
            pytest.skip("No data for test instructor")

        # Manually introduce NaN in Total_Sections before formatting
        filtered_data.loc[filtered_data.index[0], 'Total_Sections'] = np.nan

        # Format converts to long format, so Total_Sections becomes a row
        formatted_data = format_data(filtered_data)
        prepared_data = prepare_for_json(formatted_data)
        records = prepared_data.to_dict('records')

        # Should serialize without error
        json_str = json.dumps(records)
        parsed = json.loads(json_str)

        # Verify it's valid JSON and find the Total_Sections metric
        assert isinstance(parsed, list)

        # Find record where Metric is Total_Sections and Value is None
        found_null = False
        for record in parsed:
            if record.get('Metric') == 'Total_Sections' and record.get('Value') is None:
                found_null = True
                break

        assert found_null, "Should have found null Value for Total_Sections metric"

    def test_no_invalid_nan_in_json_output(self, sample_df):
        """Test that JSON output never contains the invalid 'NaN' literal"""
        # Get data with potential NaN values
        filtered_data = filter_data(
            df=sample_df,
            instructor="Papuzza, Antonio",
            terms=["Spring 2024", "Fall 2024"]
        )

        if len(filtered_data) == 0:
            pytest.skip("No data for test instructor")

        # Introduce various types of NaN
        filtered_data.loc[0, 'Response_Rate'] = np.nan
        if len(filtered_data) > 1:
            filtered_data.loc[1, 'Total_Sections'] = pd.NA

        formatted_data = format_data(filtered_data)
        prepared_data = prepare_for_json(formatted_data)
        records = prepared_data.to_dict('records')

        # Serialize to JSON
        json_str = json.dumps(records)

        # Critical assertion: no invalid NaN in JSON
        assert 'NaN' not in json_str, "JSON output contains invalid NaN literal"
        assert ': NaN' not in json_str, "JSON output contains invalid NaN value"
        assert 'NaN,' not in json_str, "JSON output contains invalid NaN literal"

        # Should be parseable
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)


class TestNanValueConstant:
    """Tests for the NAN_VALUE constant usage"""

    def test_nan_value_constant_is_string(self):
        """Test that NAN_VALUE is a string"""
        assert isinstance(NAN_VALUE, str)

    def test_nan_value_is_na(self):
        """Test that NAN_VALUE is 'NA'"""
        assert NAN_VALUE == "NA"

    def test_nan_value_used_in_filter_data(self):
        """Test that filter_data uses NAN_VALUE for missing metrics"""
        # This is more of an integration test
        # We verify that when metrics are missing, they're filled with NAN_VALUE
        from awslambda import filter_data

        # The actual test is in test_filter_data.py, but we verify the constant here
        assert NAN_VALUE == "NA"
