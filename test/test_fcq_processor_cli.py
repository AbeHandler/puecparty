"""Test the fcq_processor command line interface"""
import subprocess
import os
import pandas as pd

def test_fcq_processor_with_instructor():
    """Test that fcq_processor can filter by instructor with specific terms"""

    # Clean up any existing output file
    output_file = "fcq.Papuzza_Antonio.all.csv"
    if os.path.exists(output_file):
        os.remove(output_file)

    # Run the command
    result = subprocess.run([
        "python", "fcq_processor.py",
        "--file", "fcq.csv",
        "--instructor", "Papuzza, Antonio",
        "--terms", "Spring 2024", "Fall 2024"
    ], capture_output=True, text=True)

    # Check that the command succeeded
    assert result.returncode == 0, f"Command failed with: {result.stderr}"

    # Check that output file was created
    assert os.path.exists(output_file), "Output file was not created"

    # Load and validate the output
    df = pd.read_csv(output_file)

    # Should have data for multiple courses
    assert len(df) > 0, "Output file is empty"

    # Should have the expected columns
    assert set(df.columns) == {"Sbjct", "Crse", "Crse Title", "Metric", "Value"}

    # Should have data for Business Leadership course
    busm_courses = df[df["Crse Title"] == "Business Leadership"]
    assert len(busm_courses) > 0, "Missing expected Business Leadership course"

    # Clean up
    os.remove(output_file)

    print("✓ fcq_processor CLI test passed")

if __name__ == "__main__":
    test_fcq_processor_with_instructor()
