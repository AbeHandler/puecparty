'''
This is the main aws lambda code that serves as the backend
for the webiste
'''

import pandas as pd
import json
import re
import sys
import traceback


def validiate_mode(mode):
    if type(mode) != str:
        raise ValueError(f"Invalid mode: '{mode}'. Must be'calendar' or 'academic'")
    if mode == "calendar":
        return True
    elif mode == "academic":
        return True
    else:
        raise ValueError(f"Invalid mode: '{mode}'. Must be 'calendar' or 'academic'")
    
    return True

def get_longitudinal_scores(path="https://d2o3ke970u6qa7.cloudfront.net/scoreby_year_2020_present.csv", instructor_name="Doe, John"):
    df = pd.read_csv("https://d2o3ke970u6qa7.cloudfront.net/scoreby_year_2020_present.csv")

    # Evaluation columns (exclude non-metrics)
    eval_cols = [
        "Interact","Reflect","Connect","Collab","Contrib","Eval","Synth",
        "Diverse","Respect","Challenge","Creative","Discuss","Feedback",
        "Grading","Questions","Tech","Overall_Avg"
    ]

    # Filter for the instructor and ALL_BUSN
    df_filtered = df[df["Instructor Name"].isin([instructor_name, "ALL_BUSN"])]

    # Melt to long format
    df_long = df_filtered.melt(
        id_vars=["Year", "Instructor Name"],
        value_vars=eval_cols,
        var_name="Metric",
        value_name="Score"
    )

    # Drop missing scores
    df_long = df_long.dropna(subset=["Score"])

    # Convert to JSONL-style list of dicts
    records = [
        {
            "Year": int(row["Year"]),
            "Instructor": row["Instructor Name"],
            "Metric": row["Metric"],
            "Score": float(row["Score"])
        }
        for _, row in df_long.iterrows()
    ]

    return records


def generate_terms(start_year, end_year, mode="calendar", include_summer=False):
    """
    Generate terms based on mode and summer inclusion
    
    Args:
        start_year: Starting year
        end_year: For academic mode, this is the year of the final Spring term
        mode: "calendar" or "academic"  
        include_summer: Whether to include Summer terms
    """
    terms = []
    validiate_mode(mode)
    
    if mode == "calendar":
        # All terms within calendar years
        for year in range(start_year, end_year + 1):
            terms.append(f"Spring {year}")
            if include_summer:
                terms.append(f"Summer {year}")
            terms.append(f"Fall {year}")
            
    elif mode == "academic":
        # Academic years: Fall start_year -> Spring end_year
        current_year = start_year
        while current_year < end_year:
            terms.append(f"Fall {current_year}")
            if include_summer:
                terms.append(f"Summer {current_year + 1}")
            terms.append(f"Spring {current_year + 1}")
            current_year += 1
    
    return terms

def validate_terms(terms):
    """Validate terms in format ['Spring 2024','Fall 2024']"""

    if terms is None:
        raise ValueError(f"You must provide a valid list of terms to use the lambda")
    
    # Valid term names from the dataset
    valid_terms = {"Fall", "Spring", "Summer"}
    
    # Regex pattern: (Fall|Spring|Summer) YYYY
    pattern = r"^(Fall|Spring|Summer) (\d{4})$"
    
    validated_terms = []
    for term in terms:
        match = re.match(pattern, term)
        if not match:
            raise ValueError(f"Invalid term format: '{term}'. Expected format: 'Fall 2024'")
        
        term_name, year = match.groups()
        
        # Additional validation
        if term_name not in valid_terms:
            raise ValueError(f"Invalid term:'{term_name}'. Must be one of {valid_terms}")
        
        year_int = int(year)
        if year_int < 1900 or year_int > 2030:  # reasonable year range
            raise ValueError(f"Invalid year: {year_int}")
            
        validated_terms.append((term_name, year_int))
    
    return validated_terms

def round_cols(df, round_these_cols=None, decimals=2):
    for col in round_these_cols:
        df[col] = df[col].round(decimals)
        # format with trailing zeros
        df[col] = df[col].apply(lambda x: f"{x:.{decimals}f}" if (pd.notnull(x) and x != "NA") else x)
    return df

def _validate_df(_df):
    '''Check that the input data from S3 matches the input format'''
    assert set(_df["Term"].unique()) == set(["Fall", "Spring", "Summer"])
    assert "BUSN" in _df["College"].unique()
    assert len(_df) > 0
    COLS = {'# Resp',
            'Campus',
            'Challenge',
            'Collab',
            'College',
            'Connect',
            'Contrib',
            'Creative',
            'Crse',
            'Crse Lvl',
            'Crse Title',
            'Crse Type',
            'Dept',
            'Discuss',
            'Diverse',
            'Enroll',
            'Eval',
            'Feedback',
            'Grading',
            'Instr Grp',
            'Instructor Name',
            'Interact',
            'Questions',
            'Reflect',
            'Resp Rate',
            'Respect',
            'Sbjct',
            'Sect',
            'Synth',
            'Tech',
            'Term',
            'Term_cd',
            'Year'}
    assert set(_df.columns) == COLS, "Invalid data format"
    first_digit = _df["Crse"].astype(str).str[0].astype(int)
    bad_values = _df.loc[~((first_digit >= 1) & (first_digit <= 9)), "Crse"]
    assert ((first_digit >= 1) & (first_digit <= 9)).all(), "Found Crse values outside 1–7 range"

def load_df():
    try:
        # Use HTTPS instead of HTTP - CloudFront redirects HTTP to HTTPS
        df = pd.read_csv("https://d2o3ke970u6qa7.cloudfront.net/fcq.csv")
        
        _validate_df(df)
        # only BUSN courses are considered
        df = df[df["College"] == "BUSN"].copy()
        return df
    except Exception as e:
        error_type, error_value, tb = sys.exc_info()
        tb_info = traceback.extract_tb(tb)[-1]  # last call in the stack

        error_file = tb_info.filename
        error_line = tb_info.lineno
        error_func = tb_info.name
        error_msg = str(e)

        print(f"Error occurred in {error_file}, line {error_line}, in {error_func}: {error_msg}")
        print(traceback.format_exc())
        raise Exception(f"Failed to load CSV data: {str(e)}")

def apply_valid_for_stats_filter(df):
    """
    Apply ValidForStats logic to a dataframe.

    This determines which sections should be included in statistical calculations
    based on enrollment and course level:
    - Lower-level courses (1000-4999): enrollment >= 10
    - Graduate courses (5000+): all sections included

    Args:
        df: DataFrame with columns 'Crse' and 'Enroll'

    Returns:
        DataFrame with 'ValidForStats' column added
    """
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Assert Crse is numeric
    assert pd.api.types.is_integer_dtype(df["Crse"]) or df["Crse"].astype(str).str.isnumeric().all(), \
        "Crse column must be all integers"

    # Convert to int
    df["Crse"] = df["Crse"].astype(int)

    # Get first digit. This should be {1 ... 7}
    first_digit = df["Crse"].astype(str).str[0].astype(int)
    assert ((first_digit >= 1) & (first_digit <= 7)).all(), "Found Crse values outside 1–7 range"

    # Define ValidForStats:
    # - For undergrad courses (1000-4999): require enrollment >= 10
    # - For graduate courses (5000+): include all sections
    df["ValidForStats"] = ((df["Enroll"] >= 10) & (first_digit <= 4)) | (first_digit > 4)

    return df

def filter_data(
    df,
    instructor=None,
    course_title=None,
    terms=None,
    exclude_instructor=None,
):
    """Filter and process FCQ data"""

    if exclude_instructor is not None:
        if course_title is None:
            raise ValueError("You must specify a course to exclude an instructor")

    # terms is a list like ["Spring 2024", "Fall 2024"]
    validate_terms(terms)

    df["Year"] = df["Year"].astype(int)

    df["Term_Year"] = df["Term"] + " " + df["Year"].astype(str)

    available_instructors = df["Instructor Name"].unique().tolist()    

    filtered_df = df.copy()

    # Start with the entire dataframe and apply filters one by one
    if instructor is not None:
        # Replace assertion with proper error handling
        if instructor not in available_instructors:
            raise ValueError(f"Instructor'{instructor}' not found. Available instructors: {len(available_instructors)} total")
        
        # Filter by instructor
        filtered_df = filtered_df[filtered_df["Instructor Name"] == instructor].copy()
    
    #filter to desired terms
    filtered_df = filtered_df[filtered_df["Term_Year"].isin(terms)].copy()

    # Filter by course title if provided
    if course_title:
        available_courses = df["Crse Title"].unique().tolist()
        if course_title not in available_courses:
            raise ValueError(f"Course title'{course_title}' not found")
        filtered_df = filtered_df[filtered_df["Crse Title"] == course_title].copy()     

    if exclude_instructor:
        # this is used to compute the average over all instructors
        filtered_df = filtered_df[filtered_df["Instructor Name"] != exclude_instructor].copy()

    # Check if we have any data left after filtering
    if len(filtered_df) == 0:
        return []
    
    # Metrics we want.
    # Note: these metrics only counted when enrollment is more than 10
    metrics = [
        "Interact", "Reflect", "Connect", "Collab", "Contrib", "Eval",
        "Synth", "Diverse", "Respect", "Challenge", "Creative", "Discuss",
        "Feedback", "Grading", "Questions", "Tech",
    ]

    # Check if required columns exist
    missing_columns = [col for col in metrics if col not in filtered_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Create a section identifier to count unique sections per term
    filtered_df["Section_ID"] = filtered_df["Term_Year"] + "_" + filtered_df["Sect"].astype(str)

    # Apply ValidForStats filtering logic (shared method)
    filtered_df = apply_valid_for_stats_filter(filtered_df)

    # Aggregate metadata (count, response rate, terms) across all sections
    metadata_df = (
        filtered_df.groupby(["Sbjct", "Crse", "Crse Title"])
        .agg(
            Count=("Crse Title", "size"),  # Total number of rows/sections
            Response_Rate=("Resp Rate", "mean"),
            Terms=("Term_Year", lambda x: list(x.unique())),
        )
        .reset_index()
    )

    # Count only valid sections (matching histogram logic)
    valid_sections_df = (
        filtered_df[filtered_df["ValidForStats"]]
        .groupby(["Sbjct", "Crse", "Crse Title"])
        .agg(
            Total_Sections=("Section_ID", "nunique")
        )
        .reset_index()
    )

    # Merge metadata and section counts
    grouped_df = metadata_df.merge(valid_sections_df, on=["Sbjct", "Crse", "Crse Title"], how="left")

    # these are only for enrollment >= 10
    metrics_df = (
        filtered_df[filtered_df["ValidForStats"]]
        .groupby(["Sbjct", "Crse", "Crse Title"])
        .agg({metric: "mean" for metric in metrics})
        .reset_index()
    )

    grouped_df = grouped_df.merge(metrics_df, on=["Sbjct", "Crse", "Crse Title"], how="left")

    # Round numeric columns first (before converting to strings)
    grouped_df = round_cols(grouped_df, round_these_cols=metrics)

    # if enroll are less than 10 you will get NAs
    for metric in metrics:
        if metric in grouped_df.columns:
            grouped_df[metric] = grouped_df[metric].fillna("NA")
    
    return grouped_df

def format_result_table_long(df):
    df = df.set_index(["Sbjct", "Crse", "Crse Title"])
    long_df = df.melt(
        ignore_index=False, value_vars=df.columns, var_name="Metric", value_name="Value"
    )
    long_df = long_df.reset_index()
    long_df.sort_values(by=["Sbjct", "Crse", "Crse Title", "Metric"])
    return long_df

def prioritize_metrics_sort_corrected(df):
    custom_order = {"Count": -3, "Total_Sections": -2, "Response_Rate": -1}
    df["Metric_Order"] = df["Metric"].map(custom_order).fillna(0)
    df = df.sort_values(by=["Sbjct", "Crse", "Crse Title", "Metric_Order", "Metric"])
    df = df.drop(columns="Metric_Order")
    return df

def format_data(df):
    formatted_result = format_result_table_long(df)
    return prioritize_metrics_sort_corrected(formatted_result)

def get_section_scores_for_histogram(df, instructor, terms):
    """
    Get raw section-level scores for histogram visualization.
    Returns individual section scores (not aggregated) for each metric.

    Args:
        df: The main FCQ dataframe
        instructor: Instructor name to filter by
        terms: List of terms like ["Spring 2024", "Fall 2024"]

    Returns:
        List of dicts with section-level scores for histogram binning
    """
    # Validate inputs
    validate_terms(terms)

    df["Year"] = df["Year"].astype(int)
    df["Term_Year"] = df["Term"] + " " + df["Year"].astype(str)

    # Filter by instructor
    filtered_df = df[df["Instructor Name"] == instructor].copy()

    # Filter by terms
    filtered_df = filtered_df[filtered_df["Term_Year"].isin(terms)].copy()

    if len(filtered_df) == 0:
        return []

    # Metrics we want for histograms
    metrics = [
        "Interact", "Reflect", "Connect", "Collab", "Contrib", "Eval",
        "Synth", "Diverse", "Respect", "Challenge", "Creative", "Discuss",
        "Feedback", "Grading", "Questions", "Tech",
    ]

    # Apply ValidForStats filtering logic (shared method)
    filtered_df = apply_valid_for_stats_filter(filtered_df)

    # Only include sections valid for stats
    valid_sections = filtered_df[filtered_df["ValidForStats"]].copy()

    if len(valid_sections) == 0:
        return []

    # Build result: one record per section per metric
    results = []

    for _, row in valid_sections.iterrows():
        course_key = f"{row['Sbjct']} {row['Crse']}"
        course_title = row['Crse Title']
        term_year = row['Term_Year']
        section = row['Sect']

        for metric in metrics:
            if pd.notnull(row[metric]):
                results.append({
                    "Course": course_key,
                    "Course_Title": course_title,
                    "Term": term_year,
                    "Section": section,
                    "Metric": metric,
                    "Score": float(row[metric])
                })

    return results

def handle_longitudinal_scores_event(event):
    """Handle request for instructor + BUSN-wide evaluation scores"""
    path = event.get("path", "/tmp/scoreby_year_2020_present.csv")
    instructor = event.get("instructor")

    if not instructor:
        raise ValueError("Missing required field: 'instructor'")

    data = get_longitudinal_scores(path=path, instructor_name=instructor)
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({
            "success": True,
            "data": data,
            "count": len(data)
        })
    }

def handle_histogram_data_event(event, df):
    """Handle request for raw section-level scores for histogram visualization"""
    instructor = event.get("instructor")
    terms = event.get("terms")

    if not instructor:
        raise ValueError("Missing required field: 'instructor'")
    if not terms:
        raise ValueError("Missing required field: 'terms'")

    data = get_section_scores_for_histogram(df, instructor, terms)

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({
            "success": True,
            "data": data,
            "count": len(data)
        })
    }

def lambda_handler(event, context):
    """
    AWS Lambda handler function
    """
    try:
        print(f"Received event: {json.dumps(event)}")
        action = event.get("action")

        # Handle action-based requests
        if action is not None and action == "get_longitudinal_scores":
            return handle_longitudinal_scores_event(event)
        elif action is not None and action == "get_histogram_data":
            df = load_df()
            return handle_histogram_data_event(event, df)

        # Extract parameters for standard filter request
        instructor = event.get('instructor')
        course_title = event.get('course_title')
        terms = event.get('terms')
        exclude_instructor = event.get('exclude_instructor')


        print(f"Parameters: instructor={instructor}, course_title={course_title}, terms={terms}")

        df = load_df()

        # Call your filter function
        filtered_data = filter_data(
            df=df,
            instructor=instructor,
            course_title=course_title,
            exclude_instructor=exclude_instructor,
            terms=terms
        )

        if len(filtered_data) == 0:
            result = []
        else:
            print(f"Filtered data shape: {filtered_data.shape}")
            
            # Format the data
            formatted_data = format_data(filtered_data)
            
            print(f"Formatted data shape: {formatted_data.shape}")
            
            # Convert DataFrame to JSON-serializable format
            result = formatted_data.to_dict('records')
        
        return {
           'statusCode': 200,
           'headers': {
               'Content-Type':'application/json',
               'Access-Control-Allow-Origin':'*'
            },
           'body': json.dumps({
               'success': True,
               'data': result,
               'count': len(result)
            })
        }
        
    except Exception as e:
        # Get the full traceback for better debugging
        error_trace = traceback.format_exc()
        error_message = str(e) if str(e) else f"Unknown error of type {type(e).__name__}"
        error_type = type(e).__name__
        
        print(f"Error occurred: {error_message}")
        print(f"Error type: {error_type}")
        print(f"Full traceback: {error_trace}")
        
        # Additional debugging info
        print(f"Event received: {event}")
        print(f"Context: {context}")
        
        return {
           'statusCode': 500,
           'headers': {
               'Content-Type':'application/json',
               'Access-Control-Allow-Origin':'*'
            },
           'body': json.dumps({
               'success': False,
               'error': error_message,
               'error_type': error_type,
               'debug_info': {
                   'has_error_message': len(error_message) > 0,
                   'original_error_str': str(e),
                   'error_repr': repr(e)
                }
            })
        }