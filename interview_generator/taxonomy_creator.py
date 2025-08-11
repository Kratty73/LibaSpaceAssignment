import os
import pandas as pd
import openai
import json
from tqdm import tqdm
import time
from dotenv import load_dotenv
import ast

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

MODEL_NAME = "gpt-4o-mini"
INPUT_CSV_PATH = "extracted_skills.csv"       # <-- Input from your previous script
OUTPUT_JSON_PATH = "skill_taxonomy.json"      # <-- Final taxonomy output
CHECKPOINT_JSON_PATH = "skill_taxonomy_checkpoint.json" # <-- Checkpoint file

# --- Define the Top-Level Categories for Classification ---
SKILL_CATEGORIES = [
    "Core Programming",
    "Data Analysis & Visualization",
    "Databases & Warehousing",
    "Data Engineering & Pipelines",
    "Machine Learning & AI",
    "Cloud & DevOps",
    "Project Management & Soft Skills",
    "Business Intelligence & Reporting",
    "Other" # A catch-all for skills that don't fit
]

# --- Helper Function to Classify a Skill ---

def classify_skill(skill_name: str, retries=3, initial_delay=5) -> str:
    """
    Classifies a single skill into one of the predefined categories using an LLM.
    """
    system_prompt = f"""
    You are an expert AI assistant that categorizes technical and professional skills.
    Your task is to classify the given skill into exactly one of the following predefined categories.
    Return only the single, most appropriate category name and nothing else.

    Categories:
    {json.dumps(SKILL_CATEGORIES, indent=2)}
    """
    
    delay = initial_delay
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Classify this skill: '{skill_name}'"}
                ],
                temperature=0.0,
            )
            category = response.choices[0].message.content.strip().replace('"', '')
            
            # Validate that the response is one of the allowed categories
            if category in SKILL_CATEGORIES:
                return category
            else:
                # If the model hallucinates a new category, try to find a partial match
                for valid_cat in SKILL_CATEGORIES:
                    if valid_cat.lower() in category.lower():
                        return valid_cat
                # If no match, it's an invalid response, return "Other"
                print(f"Warning: Model returned an invalid category '{category}' for skill '{skill_name}'. Classifying as 'Other'.")
                return "Other"

        except Exception as e:
            print(f"An error occurred for skill '{skill_name}': {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2

    print(f"Failed to classify skill '{skill_name}' after all retries. Assigning to 'Other'.")
    return "Other" # Default to "Other" if all retries fail

# --- Main Execution Logic ---

if __name__ == "__main__":
    print("Starting taxonomy creation process...")

    # Load the extracted skills from the CSV
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        # Safely evaluate the string representation of lists
        df['extracted_skills'] = df['extracted_skills'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
        )
        all_skills_flat = [skill.lower() for sublist in df['extracted_skills'] for skill in sublist]
        unique_skills = sorted(list(set(all_skills_flat)))
        print(f"Loaded {len(unique_skills)} unique skills to be classified.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_CSV_PATH}")
        exit()
    except Exception as e:
        print(f"An error occurred while reading the skills file: {e}")
        exit()

    # --- Checkpointing and Resuming Logic ---
    taxonomy = {category: [] for category in SKILL_CATEGORIES}
    processed_skills = set()

    if os.path.exists(CHECKPOINT_JSON_PATH):
        print(f"Checkpoint file found. Resuming from {CHECKPOINT_JSON_PATH}")
        with open(CHECKPOINT_JSON_PATH, 'r') as f:
            taxonomy = json.load(f)
        # Get the set of skills that have already been processed
        for category_skills in taxonomy.values():
            processed_skills.update(category_skills)
        print(f"Resuming with {len(processed_skills)} skills already classified.")

    # Filter out skills that have already been processed
    skills_to_process = [skill for skill in unique_skills if skill not in processed_skills]
    
    # --- Main Classification Loop ---
    progress_bar = tqdm(skills_to_process, desc="Classifying Skills")
    for i, skill in enumerate(progress_bar):
        category = classify_skill(skill)
        if category not in taxonomy:
            taxonomy[category] = [] # Should not happen, but as a safeguard
        taxonomy[category].append(skill)

        # Save checkpoint periodically
        if (i + 1) % 50 == 0:
            with open(CHECKPOINT_JSON_PATH, 'w') as f:
                json.dump(taxonomy, f, indent=4)

    # --- Finalization ---
    # Final save of the completed taxonomy
    with open(OUTPUT_JSON_PATH, 'w') as f:
        # Sort skills within each category for clean output
        for category in taxonomy:
            taxonomy[category] = sorted(list(set(taxonomy[category])))
        json.dump(taxonomy, f, indent=4)
    
    print(f"\nTaxonomy creation complete. Saved to {OUTPUT_JSON_PATH}")
    
    # Optional: Clean up the checkpoint file
    if os.path.exists(CHECKPOINT_JSON_PATH):
        os.remove(CHECKPOINT_JSON_PATH)
