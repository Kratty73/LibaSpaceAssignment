import os
import pandas as pd
import openai
import json
from tqdm import tqdm
import time
from dotenv import load_dotenv
import re

# --- Configuration ---
# Load environment variables from a .env file
load_dotenv()

# Your API key should be in a .env file in the same directory:
# OPENAI_API_KEY="sk-..."
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("OpenAI API key not found. Please create a .env file and add your OPENAI_API_KEY.")
    exit()

client = openai.OpenAI(api_key=api_key)

# To handle Parquet files, you might need to install the pyarrow library:
# pip install pyarrow
MODEL_NAME = "gpt-4o-mini" # Switched to the more cost-effective model
INPUT_PARQUET_PATH = "jd.parquet" # <-- CHANGE THIS to your input file
OUTPUT_CSV_PATH = "extracted_skills.csv"           # <-- Output will now be a CSV file
CHECKPOINT_INTERVAL = 25 # Save progress every 25 rows

# --- Helper Function to Call OpenAI API ---

def extract_skills_from_description(description: str, retries=3, initial_delay=5) -> list:
    """
    Extracts skills from a job description using the specified OpenAI model.
    Includes exponential backoff for retries and robust JSON parsing.

    Args:
        description: The text of the job description.
        retries: Number of times to retry the API call in case of failure.
        initial_delay: The initial delay in seconds for retries.

    Returns:
        A list of extracted skills, or None if extraction fails permanently.
    """
    if not isinstance(description, str) or not description.strip():
        return []
        
    system_prompt = """
    You are an expert AI assistant specializing in technical recruitment. Your task is to analyze a job description and extract all relevant skills.
    Focus on specific, tangible skills, tools, and methodologies.
    Return the output ONLY as a single, flat JSON array of strings.
    Example: ["Python", "React", "Node.js", "Leadership"]
    Do not include any other text, explanations, or formatting outside of the JSON array.
    """
    
    delay = initial_delay
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": description}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            
            # --- Improved Robust JSON Parsing Logic ---
            try:
                # First, try to parse the content directly as valid JSON
                result_json = json.loads(content)
                if isinstance(result_json, dict):
                    for key, value in result_json.items():
                        if isinstance(value, list):
                            return value
                elif isinstance(result_json, list):
                    return result_json
            except json.JSONDecodeError:
                # If direct parsing fails, try to find and clean a JSON-like string
                match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
                if match:
                    json_string = match.group(0)
                    # If it's a set-like object `{}`, convert to a list `[]` for parsing
                    if json_string.startswith('{'):
                        json_string = '[' + json_string[1:-1] + ']'
                    
                    # Try parsing the cleaned string
                    cleaned_result = json.loads(json_string)
                    if isinstance(cleaned_result, list):
                        return cleaned_result

            # If all parsing attempts fail, raise an error to trigger a retry
            raise json.JSONDecodeError("Could not parse a valid list from the response.", content, 0)

        except openai.RateLimitError as e:
            print(f"Rate limit error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2
        except json.JSONDecodeError as e:
             print(f"JSONDecodeError on attempt {attempt + 1}: {e}. Response: {content}")
             if attempt < retries - 1:
                 time.sleep(delay)
                 delay *= 2
             else:
                 return None # Use None to signify a persistent failure
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2

    return None # Use None to signify a persistent failure after all retries

# --- Main Execution Logic ---

if __name__ == "__main__":
    print("Starting skill extraction process...")

    # Load your DataFrame from a Parquet file
    try:
        df = pd.read_parquet(INPUT_PARQUET_PATH)
        print(f"Successfully loaded {len(df)} rows from {INPUT_PARQUET_PATH}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_PARQUET_PATH}")
        exit()
    except Exception as e:
        print(f"An error occurred while reading the Parquet file: {e}")
        exit()


    # --- Filtering and Sampling Logic ---
    print("\n--- Filtering and Sampling ---")
    roles_to_sample = {'Data Analyst': 750, 'Data Engineer': 750, 'Data Science': 750}
    sampled_dfs = []
    
    # Check if 'Primary Keyword' column exists
    if 'Primary Keyword' not in df.columns:
        print(f"Error: The required column 'Primary Keyword' was not found in your Parquet file.")
        print(f"Available columns are: {list(df.columns)}")
        exit()

    for role, n_samples in roles_to_sample.items():
        # Filter for exact, case-sensitive match
        role_df = df[df['Primary Keyword'] == role]
        print(f"Found {len(role_df)} entries for role: '{role}'")
        
        if len(role_df) >= n_samples:
            sampled_dfs.append(role_df.sample(n=n_samples, random_state=42))
            print(f" -> Sampled {n_samples} entries.")
        else:
            print(f" -> Warning: Not enough entries. Using all {len(role_df)} available entries.")
            if not role_df.empty:
                sampled_dfs.append(role_df)
    
    if sampled_dfs:
        df = pd.concat(sampled_dfs).reset_index(drop=True)
        print(f"\nSuccessfully created a final dataset with {len(df)} entries.")
    else:
        print("\nError: Could not create the desired dataset. No matching roles found.")
        exit()

    # --- Checkpointing and Resuming Logic ---
    print("\n--- API Processing ---")
    start_index = 0
    if os.path.exists(OUTPUT_CSV_PATH):
        print(f"Output file found at {OUTPUT_CSV_PATH}. Resuming from last checkpoint.")
        try:
            df_processed = pd.read_csv(OUTPUT_CSV_PATH)
            # Find the last valid index that was processed
            last_processed_index = df_processed['extracted_skills'].last_valid_index()
            if last_processed_index is not None:
                start_index = last_processed_index + 1
                # Copy the already processed skills to the main dataframe
                df['extracted_skills'] = df_processed['extracted_skills']
            print(f"Resuming from index {start_index}.")
        except pd.errors.EmptyDataError:
            print("Output file is empty. Starting from scratch.")
            df['extracted_skills'] = pd.Series([[] for _ in range(len(df))], dtype=object)
    else:
        # If no output file exists, create the new column and fill with empty lists
        df['extracted_skills'] = pd.Series([[] for _ in range(len(df))], dtype=object)

    # --- Main Processing Loop ---
    for index, row in tqdm(df.iterrows(), total=len(df), initial=start_index, desc="Extracting Skills"):
        if index < start_index:
            continue

        # Get the skills
        skills = extract_skills_from_description(row['Long Description'])
        
        if skills is not None:
            df.at[index, 'extracted_skills'] = skills
        else:
            # If processing fails permanently, log a warning and assign an empty list.
            # This marks the row as "processed" so we don't retry it on the next run.
            print(f"\nWarning: Failed to process index {index} after all retries. Skipping and assigning empty list.")
            df.at[index, 'extracted_skills'] = []

        # Save progress at specified intervals
        if (index + 1) % CHECKPOINT_INTERVAL == 0:
            df.to_csv(OUTPUT_CSV_PATH, index=False)

    # Final save
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nProcessing complete. Final results saved to {OUTPUT_CSV_PATH}")

    # --- Post-processing ---
    print("\n--- Final Analysis ---")
    # Reload the final saved data to ensure consistency
    final_df = pd.read_csv(OUTPUT_CSV_PATH)
    # Convert string representation of list back to actual list
    final_df['extracted_skills'] = final_df['extracted_skills'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [])
    
    all_skills_flat = [skill for sublist in final_df['extracted_skills'] for skill in sublist]
    unique_skills = sorted(list(set(all_skills_flat)))

    print(f"Total skills extracted (including duplicates): {len(all_skills_flat)}")
    print(f"Total unique skills found: {len(unique_skills)}")
    print("\nSample of unique skills:")
    print(unique_skills[:20])