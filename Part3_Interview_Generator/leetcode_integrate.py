import os
import csv
import json

# --- CONFIGURATION ---

# 1. The CSV file containing your LeetCode data.
LEETCODE_DATA_FILE = "leetcode_dataset - lc.csv"

# 2. The JSON file to read from and append to.
#    This should be the same output file as your GitHub script.
OUTPUT_JSON_FILE = "questions_output.json"

# 3. Standardized rubric for all LeetCode problems (no LLM needed).
GENERIC_CODING_RUBRIC = {
    "Correctness": {
        "Poor": "Fails on basic test cases or has significant logical errors.",
        "Good": "Passes most test cases but may fail on edge cases.",
        "Excellent": "Solution is fully correct, handles all edge cases, and passes all test cases."
    },
    "Time Complexity": {
        "Poor": "Brute-force or significantly suboptimal (e.g., O(n^2) when O(n log n) is expected).",
        "Good": "Meets the expected time complexity for a standard approach.",
        "Excellent": "Achieves optimal time complexity, possibly using a clever or advanced technique."
    },
    "Space Complexity": {
        "Poor": "Uses excessive extra space unnecessarily.",
        "Good": "Uses a reasonable amount of extra space, in line with standard solutions.",
        "Excellent": "Achieves optimal space complexity, possibly solving the problem in-place."
    },
    "Code Quality": {
        "Poor": "Code is hard to read, poorly formatted, with meaningless variable names.",
        "Good": "Code is readable with clear variable names and some comments.",
        "Excellent": "Code is clean, well-structured, follows best practices, and is easy to understand."
    }
}


def main():
    """
    Reads LeetCode data from a CSV, combines it with existing data from a JSON file,
    and saves the result, avoiding duplicates.
    """
    # Step 1: Load existing data from the output JSON file.
    existing_data = []
    if os.path.exists(OUTPUT_JSON_FILE):
        try:
            with open(OUTPUT_JSON_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"✅ Successfully loaded {len(existing_data)} questions from '{OUTPUT_JSON_FILE}'.")
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"⚠️ Could not read or decode '{OUTPUT_JSON_FILE}'. Starting with an empty list.")
    else:
        print(f"'{OUTPUT_JSON_FILE}' not found. A new file will be created.")

    # Create a set of existing LeetCode titles for efficient duplicate checking.
    existing_leetcode_titles = {
        q['question_title'] for q in existing_data if q.get('source_file') == 'LeetCode'
    }

    # Step 2: Read the LeetCode CSV file.
    if not os.path.exists(LEETCODE_DATA_FILE):
        print(f"❌ Error: LeetCode data file not found at '{LEETCODE_DATA_FILE}'. Exiting.")
        return

    new_questions_added = 0
    try:
        with open(LEETCODE_DATA_FILE, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                title = row.get("title")
                if not title:
                    continue  # Skip rows without a title.

                # Step 3: Check for duplicates before processing.
                if title in existing_leetcode_titles:
                    continue  # Skip this problem as it's already in the JSON file.

                # Transform the new LeetCode problem into our standard format.
                transformed_data = {
                    "type": "Coding",
                    "difficulty": row.get("difficulty", "Unknown").capitalize(),
                    "evaluation_rubric": GENERIC_CODING_RUBRIC,
                    "source_file": "LeetCode",
                    "question_title": title,
                    "url": row.get("url"),
                    "acceptance_rate": row.get("acceptance_rate"),
                    "related_topics": row.get("related_topics"),
                    "companies": row.get("companies"),
                    "description": row.get("description")
                }
                existing_data.append(transformed_data)
                existing_leetcode_titles.add(title) # Add to set to prevent duplicates from within the CSV
                new_questions_added += 1
                print(f"   ➕ Added LeetCode problem: \"{title}\"")

    except Exception as e:
        print(f"❌ An error occurred while processing '{LEETCODE_DATA_FILE}': {e}")
        return

    # Step 4: Write the combined data back to the JSON file.
    if new_questions_added > 0:
        try:
            with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4)
            print(f"\n✨ Success! Added {new_questions_added} new LeetCode questions.")
            print(f"'{OUTPUT_JSON_FILE}' now contains a total of {len(existing_data)} questions.")
        except Exception as e:
            print(f"❌ An error occurred while writing to '{OUTPUT_JSON_FILE}': {e}")
    else:
        print("\n✨ No new LeetCode problems to add.")


if __name__ == "__main__":
    main()