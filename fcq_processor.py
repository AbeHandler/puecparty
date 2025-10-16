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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and group FCQ dataset.")
    parser.add_argument("--file", type=str, help="Path to the Excel file", required=True)
    parser.add_argument("--instructor", type=str, help="Instructor name to filter by", default=None)
    parser.add_argument("--exclude_instructor", type=str, help="Instructor to exclude", default=None)
    parser.add_argument("--course", type=str, help="Course title to filter by", default=None)
    parser.add_argument("--include_summer", action="store_true", help="Include Summer terms")
    
    # Term specification options
    term_group = parser.add_mutually_exclusive_group(required=True)
    term_group.add_argument("--terms", nargs="+", help="Specific terms like 'Fall 2024' 'Spring 2025'")
    term_group.add_argument("--year_range", nargs=2, type=int, metavar=("START", "END"), 
                           help="Year range like 2023 2025 (generates calendar year terms)")
    term_group.add_argument("--academic_year", nargs=2, type=int, metavar=("START", "END"),
                           help="Academic year range like 2023 2024 (Fall 2023 -> Spring 2025)")
    
    parser.add_argument("--download", action="store_true", help="Download the Excel file")
    
    args = parser.parse_args()
    
    if args.download:
        download_excel_and_upload_to_s3()
    
    # Generate terms based on input method
    if args.terms:
        terms = args.terms
    elif args.year_range:
        terms = generate_terms(args.year_range[0], args.year_range[1], "calendar", args.include_summer)
    elif args.academic_year:
        terms = generate_terms(args.academic_year[0], args.academic_year[1], "academic", args.include_summer)
    
    print(f"Using terms: {terms}")
    
    # Call the refactored filter_data function
    result = filter_data(
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