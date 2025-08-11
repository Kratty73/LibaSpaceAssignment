import os
import json
import ijson # Import the new library
from openai import OpenAI
from dotenv import load_dotenv

# --- CONFIGURATION ---

# 1. Load API Key from your .env file
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 2. Input and Output file paths
HR_DATA_FILE = "hr_interview_questions_dataset.json"
OUTPUT_JSON_FILE = "questions_output.json"

# 3. Number of new HR questions to process and add in a single run
NUM_QUESTIONS_TO_ADD = 100

# 4. Model for generating the evaluation rubric
MODEL_NAME = "gpt-4o-mini"


def generate_behavioral_rubric(question_data):
    """
    Uses an LLM to generate an evaluation rubric for a behavioral question.
    """
    question = question_data.get("question")
    ideal_answer = question_data.get("ideal_answer")
    keywords = ", ".join(question_data.get("keywords", []))

    system_prompt = """
    You are an expert HR hiring manager. Your task is to create a detailed evaluation rubric for a behavioral interview question.
    Based on the provided question, ideal answer, and keywords, generate a single JSON object for the key 'evaluation_rubric'.
    The rubric's criteria should be based on the STAR method (Situation, Task, Action, Result) and how well the candidate's answer aligns with the provided keywords.
    """
    user_prompt = f"""
    Generate the evaluation rubric JSON object for the following:

    - Question: "{question}"
    - Snippet of an Ideal Answer: "{ideal_answer}"
    - Keywords to Listen For: "{keywords}"
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("evaluation_rubric")
    except Exception as e:
        print(f"❌ LLM API call failed for question '{question[:30]}...': {e}")
        return None

def main():
    """
    Reads HR questions, generates rubrics using an LLM, and appends them
    to the main output file, avoiding duplicates.
    """
    # Step 1: Load existing data from the output JSON file
    existing_data = []
    if os.path.exists(OUTPUT_JSON_FILE):
        try:
            with open(OUTPUT_JSON_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"✅ Loaded {len(existing_data)} questions from '{OUTPUT_JSON_FILE}'.")
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"⚠️ Could not read '{OUTPUT_JSON_FILE}'. Starting with an empty list.")
    else:
        print(f"'{OUTPUT_JSON_FILE}' not found. A new file will be created.")

    # Create a set of existing HR question titles for efficient duplicate checking
    existing_hr_titles = {q['question_title'] for q in existing_data if q.get('source_file') == 'HR Dataset'}

    # Step 2: Stream the source HR dataset instead of loading it all at once
    if not os.path.exists(HR_DATA_FILE):
        print(f"❌ Error: HR data file not found at '{HR_DATA_FILE}'. Exiting.")
        return
        
    new_questions_added = 0
    print(f"⚙️  Streaming questions from '{HR_DATA_FILE}'...")
    with open(HR_DATA_FILE, 'r', encoding='utf-8') as f:
        # The 'item' prefix assumes the JSON is a list of objects at the top level: `[{...}, {...}]`
        hr_questions_stream = ijson.items(f, 'item')
        
        for question_data in hr_questions_stream:
            if new_questions_added >= NUM_QUESTIONS_TO_ADD:
                print(f"\nReached the limit of {NUM_QUESTIONS_TO_ADD} new questions for this run.")
                break

            question_title = question_data.get("question")
            if not question_title or question_title in existing_hr_titles:
                continue # Skip if no title or if it's a duplicate

            print(f"\nProcessing new HR question: \"{question_title[:60]}...\"")
            rubric = generate_behavioral_rubric(question_data)

            if rubric:
                transformed_data = {
                    "type": question_data.get("category", "Behavioral"),
                    "difficulty": question_data.get("difficulty", "Medium"),
                    "evaluation_rubric": rubric,
                    "source_file": "HR Dataset",
                    "question_title": question_title,
                    "role": question_data.get("role"),
                    "experience": question_data.get("experience"),
                    "keywords": question_data.get("keywords")
                }
                existing_data.append(transformed_data)
                existing_hr_titles.add(question_title)
                new_questions_added += 1
                print(f"   ✅ Successfully processed and added. ({new_questions_added}/{NUM_QUESTIONS_TO_ADD})")

    # Step 3: Write the final combined data back to the file
    if new_questions_added > 0:
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4)
        print(f"\n✨ Success! Added {new_questions_added} new HR questions.")
        print(f"'{OUTPUT_JSON_FILE}' now contains a total of {len(existing_data)} questions.")
    else:
        print("\n✨ No new HR questions to add from the dataset.")


if __name__ == "__main__":
    main()