'''

This is a front end for the awslambda.py used for local testing

It also contains some local util functions

'''
import argparse
import pandas as pd
import os
import time
import requests

from awslambda import filter_data
from awslambda import format_data
from awslambda import generate_terms
from awslambda import get_busn_metrics_stats
from awslambda import get_instructor_z_scores
from awslambda import get_section_z_scores

def get_scores_by_year():
    # Load data
    df = pd.read_csv("/tmp/fcq.csv")

    # Filter to BUSN
    df_busn = df[df["College"] == "BUSN"]

    # Evaluation columns
    eval_cols = [
        "Interact","Reflect","Connect","Collab","Contrib","Eval","Synth",
        "Diverse","Respect","Challenge","Creative","Discuss","Feedback",
        "Grading","Questions","Tech"
    ]

    # --- Instructor-level yearly averages ---
    df_instructor = (
        df_busn.groupby(["Year", "Instructor Name"])[eval_cols]
        .mean()
        .reset_index()
    )

    # Add per-instructor overall average
    df_instructor["Overall_Avg"] = df_instructor[eval_cols].mean(axis=1)

    # --- School-wide yearly averages ---
    df_school = (
        df_busn.groupby("Year")[eval_cols]
        .mean()
        .reset_index()
    )
    df_school["Instructor Name"] = "ALL_BUSN"
    df_school["Overall_Avg"] = df_school[eval_cols].mean(axis=1)

    # --- Combine ---
    df_combined = pd.concat([df_instructor, df_school], ignore_index=True)

    # Save to CSV
    df_combined.to_csv("/tmp/scoreby_year_2020_present.csv", index=False)

    cmd = 'aws s3 cp /tmp/scoreby_year_2020_present.csv s3://ucbfcqs/scoreby_year_2020_present.csv'
    os.system(cmd)


def get_unique_instructors():
    '''
    This is just a simple helper that can be used to populate the instructor
    drop down in index.html. For now I just copy it over into the HTML by hand.
    For now, it will need to be rerun each semester as new instructors are added to Leeds
    '''
    df = pd.read_csv('/tmp/fcq.csv')
    df_busn = df[df['College'] == 'BUSN']
    unique_instructors = df_busn['Instructor Name'].unique()

    with open("instructors.txt", 'w') as of:
        of.write(f'<datalist id="instructorList">\n')
        for line in unique_instructors:
            name = line.strip()
            of.write(f'  <option value="{name}">\n')
        of.write("</datalist>")

def download_excel_and_upload_to_s3():
    '''
    This preprocesses data for S3 allowing the lambda to run much more quickly
    '''

    URL = "https://www.colorado.edu/fcq/media/42"
    FILE_PATH = "/tmp/fcq.xlsx"
    CSV_PATH = "/tmp/fcq.csv"
    MAX_AGE = 24 * 3600  # 24 hours in seconds

    r = requests.get(URL)
    r.raise_for_status()
    with open(FILE_PATH, "wb") as f:
        f.write(r.content)
    print("Download complete.")

    df = pd.read_excel(FILE_PATH, sheet_name="FCQ Results", skiprows=6)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved CSV to {CSV_PATH}")
    os.system("aws s3 cp /tmp/fcq.csv s3://ucbfcqs/fcq.csv")

def ensure_csv():

    CSV_PATH = "/tmp/fcq.csv"
    URL = "https://ucbfcqs.s3.us-east-1.amazonaws.com/fcq.csv"

    if not os.path.exists(CSV_PATH):
        print(f"{CSV_PATH} not found. Downloading from {URL}...")
        r = requests.get(URL)
        r.raise_for_status()
        with open(CSV_PATH, "wb") as f:
            f.write(r.content)
        print("Download complete.")
    else:
        print(f"{CSV_PATH} already exists.")


def download_excel_and_upload_to_s3():
    '''
    This preprocesses data for S3 allowing the lambda to run much more quickly
    '''
    URL = "https://www.colorado.edu/fcq/media/42"
    FILE_PATH = "/tmp/fcq.xlsx"
    CSV_PATH = "/tmp/fcq.csv"

    r = requests.get(URL)
    r.raise_for_status()
    with open(FILE_PATH, "wb") as f:
        f.write(r.content)
    print("Download complete.")

    df = pd.read_excel(FILE_PATH, sheet_name="FCQ Results", skiprows=6)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved CSV to {CSV_PATH}")
    os.system("aws s3 cp /tmp/fcq.csv s3://ucbfcqs/fcq.csv")


def demo_busn_metrics_stats():
    '''
    Demo function to show how to use get_busn_metrics_stats.

    This calculates mean and standard deviation for each evaluation metric
    across all BUSN instructors during a specified time period.
    '''
    print("\n=== Demo: BUSN Metrics Statistics ===\n")

    # Ensure CSV is available
    ensure_csv()

    # Load the dataframe
    df = pd.read_csv("/tmp/fcq.csv")

    # Filter to BUSN only (as done in the lambda)
    df = df[df["College"] == "BUSN"].copy()
    print(f"Loaded {len(df)} BUSN records")

    # Define terms to analyze
    terms = ["Fall 2022", "Spring 2023", "Fall 2023", "Spring 2024"]
    print(f"Analyzing terms: {terms}\n")

    # Get BUSN-wide statistics
    stats = get_busn_metrics_stats(df, terms)

    # Display results
    print(f"Found statistics for {len(stats)} metrics:\n")
    print(f"{'Metric':<15} {'Mean':<8} {'Std':<8} {'Count':<8}")
    print("-" * 45)

    for metric_stat in stats:
        print(f"{metric_stat['Metric']:<15} "
              f"{metric_stat['Mean']:<8.2f} "
              f"{metric_stat['Std']:<8.2f} "
              f"{metric_stat['Count']:<8}")

    print("\n=== Demo Complete ===\n")


def demo_instructor_z_scores():
    '''
    Demo function to show how to use get_instructor_z_scores.

    This calculates z-scores (standard deviations from mean) for an instructor's
    metrics compared to all BUSN instructors during a specified time period.

    A positive z-score means the instructor is above the BUSN average.
    A negative z-score means the instructor is below the BUSN average.
    '''
    print("\n=== Demo: Instructor Z-Scores ===\n")

    # Ensure CSV is available
    ensure_csv()

    # Load the dataframe
    df = pd.read_csv("/tmp/fcq.csv")

    # Filter to BUSN only (as done in the lambda)
    df = df[df["College"] == "BUSN"].copy()
    print(f"Loaded {len(df)} BUSN records")

    # Define instructor and terms to analyze
    instructor = "Handler, Abram"
    terms = ["Fall 2022", "Spring 2023", "Fall 2023", "Spring 2024"]
    print(f"Analyzing instructor: {instructor}")
    print(f"Time period: {terms}\n")

    # Get z-scores
    z_scores = get_instructor_z_scores(df, instructor, terms)

    # Display results
    print(f"Found z-scores for {len(z_scores)} metrics:\n")
    print(f"{'Metric':<12} {'Instructor':<12} {'BUSN Mean':<12} {'BUSN Std':<10} {'Z-Score':<10} {'Interpretation'}")
    print("-" * 90)

    for metric_stat in z_scores:
        z = metric_stat['Z_Score']

        # Interpret the z-score
        if abs(z) < 0.5:
            interpretation = "At mean"
        elif z >= 2.0:
            interpretation = "Significantly above (top ~2.5%)"
        elif z >= 1.0:
            interpretation = "Above average"
        elif z >= 0.5:
            interpretation = "Slightly above"
        elif z <= -2.0:
            interpretation = "Significantly below (bottom ~2.5%)"
        elif z <= -1.0:
            interpretation = "Below average"
        else:  # z <= -0.5
            interpretation = "Slightly below"

        print(f"{metric_stat['Metric']:<12} "
              f"{metric_stat['Instructor_Score']:<12.2f} "
              f"{metric_stat['BUSN_Mean']:<12.2f} "
              f"{metric_stat['BUSN_Std']:<10.2f} "
              f"{z:<10.2f} "
              f"{interpretation}")

    print("\n=== Demo Complete ===\n")
    print("Interpretation guide:")
    print("  Z-Score > 2.0:  Significantly above average (top ~2.5%)")
    print("  Z-Score > 1.0:  Above average")
    print("  Z-Score > 0.5:  Slightly above average")
    print("  |Z-Score| < 0.5: At the mean")
    print("  Z-Score < -0.5: Slightly below average")
    print("  Z-Score < -1.0: Below average")
    print("  Z-Score < -2.0: Significantly below average (bottom ~2.5%)")
    print()


def demo_section_z_scores():
    '''
    Demo function to show how to use get_section_z_scores.

    This calculates z-scores for EVERY section and EVERY metric compared to
    BUSN-wide statistics during a specified time period.

    Unlike instructor z-scores which aggregate by instructor, this gives you
    granular section-level data showing how each individual section performed
    relative to all BUSN sections.
    '''
    print("\n=== Demo: Section-Level Z-Scores ===\n")

    # Ensure CSV is available
    ensure_csv()

    # Load the dataframe
    df = pd.read_csv("/tmp/fcq.csv")

    # Filter to BUSN only (as done in the lambda)
    df = df[df["College"] == "BUSN"].copy()
    print(f"Loaded {len(df)} BUSN records")

    # Define terms to analyze
    terms = ["Fall 2021", "Spring 2022", "Fall 2022", "Spring 2023", "Fall 2023", "Spring 2024"]
    print(f"Analyzing terms: {terms}")

    # Option 1: Get z-scores for a specific instructor
    instructor = "Papuzza, Antonio"
    print(f"Filtering for instructor: {instructor}\n")

    # Get section-level z-scores
    z_scores = get_section_z_scores(df, terms, instructor=instructor)

    # Display results
    print(f"Found z-scores for {len(z_scores)} section-metric combinations displaying z scores below 3\n")

    z_scores = [o for o in z_scores if o["Z_Score"] < 2 or o["Z_Score"] > 2]

    # Print header
    print(f"{'Course':<12} {'Term':<15} {'Section':<6} {'Metric':<12} "
          f"{'Section_Score':<14} {'BUSN_Mean':<12} {'Z_Score':<10} {'Interpretation'}")
    print("-" * 90)

    for i, result in enumerate(z_scores):
        z = result['Z_Score']

        # Interpret the z-score
        if abs(z) < 0.5:
            interpretation = "At mean"
        elif z >= 2.0:
            interpretation = "Significantly above"
        elif z >= 1.0:
            interpretation = "Above average"
        elif z >= 0.5:
            interpretation = "Slightly above"
        elif z <= -2.0:
            interpretation = "Significantly below"
        elif z <= -1.0:
            interpretation = "Below average"
        else:  # z <= -0.5
            interpretation = "Slightly below"

        if z < -3:
            print(f"{result['Course']:<12} "
                  f"{result['Term']:<15} "
                  f"{result['Section']:<6} "
                  f"{result['Metric']:<12} "
                  f"{result['Section_Score']:<14.2f} "
                  f"{result['BUSN_Mean']:<12.2f} "
                  f"{result['Z_Score']:<10.2f} "
                  f"{interpretation}")

    # Summary statistics
    import numpy as np
    all_z_scores = [r['Z_Score'] for r in z_scores]
    print(f"\nSummary of all z-scores:")
    print(f"  Mean: {np.mean(all_z_scores):.2f}")
    print(f"  Median: {np.median(all_z_scores):.2f}")
    print(f"  Std Dev: {np.std(all_z_scores):.2f}")
    print(f"  Min: {np.min(all_z_scores):.2f}")
    print(f"  Max: {np.max(all_z_scores):.2f}")

    # Distribution breakdown
    significantly_above = sum(1 for z in all_z_scores if z >= 2.0)
    above = sum(1 for z in all_z_scores if 1.0 <= z < 2.0)
    slightly_above = sum(1 for z in all_z_scores if 0.5 <= z < 1.0)
    at_mean = sum(1 for z in all_z_scores if -0.5 < z < 0.5)
    slightly_below = sum(1 for z in all_z_scores if -1.0 < z <= -0.5)
    below = sum(1 for z in all_z_scores if -2.0 < z <= -1.0)
    significantly_below = sum(1 for z in all_z_scores if z <= -2.0)

    total = len(all_z_scores)
    print(f"\nDistribution of z-scores:")
    print(f"  Significantly above (>= 2.0):  {significantly_above:4d} ({100*significantly_above/total:5.1f}%)")
    print(f"  Above average (1.0-2.0):        {above:4d} ({100*above/total:5.1f}%)")
    print(f"  Slightly above (0.5-1.0):       {slightly_above:4d} ({100*slightly_above/total:5.1f}%)")
    print(f"  At mean (-0.5 to 0.5):          {at_mean:4d} ({100*at_mean/total:5.1f}%)")
    print(f"  Slightly below (-1.0 to -0.5):  {slightly_below:4d} ({100*slightly_below/total:5.1f}%)")
    print(f"  Below average (-2.0 to -1.0):   {below:4d} ({100*below/total:5.1f}%)")
    print(f"  Significantly below (<= -2.0):  {significantly_below:4d} ({100*significantly_below/total:5.1f}%)")

    print("\n=== Demo Complete ===\n")
    print("Note: To get z-scores for ALL instructors (not just one), call:")
    print("  get_section_z_scores(df, terms, instructor=None)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and group FCQ dataset.")
    parser.add_argument("--file", type=str, help="Path to the Excel file", required=False)
    parser.add_argument("--instructor", type=str, help="Instructor name to filter by", default=None)
    parser.add_argument("--exclude_instructor", type=str, help="Instructor to exclude", default=None)
    parser.add_argument("--course", type=str, help="Course title to filter by", default=None)
    parser.add_argument("--include_summer", action="store_true", help="Include Summer terms")

    # Term specification options
    term_group = parser.add_mutually_exclusive_group(required=False)
    term_group.add_argument("--terms", nargs="+", help="Specific terms like 'Fall 2024' 'Spring 2025'")
    term_group.add_argument("--year_range", nargs=2, type=int, metavar=("START", "END"),
                           help="Year range like 2023 2025 (generates calendar year terms)")
    term_group.add_argument("--academic_year", nargs=2, type=int, metavar=("START", "END"),
                           help="Academic year range like 2023 2024 (Fall 2023 -> Spring 2025)")

    parser.add_argument("--download", action="store_true", help="Download the Excel file")
    parser.add_argument("--demo_stats", action="store_true", help="Run demo of BUSN metrics statistics")
    parser.add_argument("--demo_z_scores", action="store_true", help="Run demo of instructor z-scores")
    parser.add_argument("--demo_section_z_scores", action="store_true", help="Run demo of section-level z-scores")

    args = parser.parse_args()

    if args.download:
        download_excel_and_upload_to_s3()
        exit(0)

    if args.demo_stats:
        demo_busn_metrics_stats()
        exit(0)

    if args.demo_z_scores:
        demo_instructor_z_scores()
        exit(0)

    if args.demo_section_z_scores:
        demo_section_z_scores()
        exit(0)

    # Require file and terms for normal operation
    if not args.file:
        parser.error("--file is required unless using --download, --demo_stats, --demo_z_scores, or --demo_section_z_scores")

    if not (args.terms or args.year_range or args.academic_year):
        parser.error("One of --terms, --year_range, or --academic_year is required")

    # Generate terms based on input method
    if args.terms:
        terms = args.terms
    elif args.year_range:
        terms = generate_terms(args.year_range[0], args.year_range[1], "calendar", args.include_summer)
    elif args.academic_year:
        terms = generate_terms(args.academic_year[0], args.academic_year[1], "academic", args.include_summer)

    print(f"Using terms: {terms}")

    # Load the dataframe
    print(f"Loading data from {args.file}...")
    df = pd.read_csv(args.file)

    # Filter to BUSN only
    df = df[df["College"] == "BUSN"].copy()
    print(f"Loaded {len(df)} BUSN records")

    # Call the refactored filter_data function
    result = filter_data(
        df=df,
        instructor=args.instructor,
        course_title=args.course,
        terms=terms,
        exclude_instructor=args.exclude_instructor
    )

    if len(result) == 0:
        print("No data found with the specified filters.")
        exit(1)

    result = format_data(result)

    # Generate output filename
    instructor_part = args.instructor.replace(" ", "_").replace(",", "") if args.instructor else "all"
    course_part = args.course.replace(" ", "_") if args.course else "all"
    outfile = f"fcq.{instructor_part}.{course_part}.csv"

    result.to_csv(outfile, index=False)
    print(f"Wrote {len(result)} rows to {outfile}")