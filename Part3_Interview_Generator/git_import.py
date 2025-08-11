import os
import requests
import json
import re
from openai import OpenAI
import base64
from dotenv import load_dotenv

# --- CONFIGURATION ---

# 1. Load API Key from your .env file
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 2. Your GitHub repository URL
GITHUB_REPO_URL = "https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers" 

# 3. Model Selection ('gpt-4o-mini' is fast/cheap, 'gpt-4o' is higher quality)
MODEL_NAME = "gpt-4o-mini" 

# 4. The pattern to separate individual questions WITHIN the Q&A section.
#    This pattern is set to split before a heading like '### Q1:'.
QUESTION_DELIMITER_PATTERN = r'\n(?=###\s*Q\d+:)'

# 5. The heading that marks the beginning of the actual Q&A content.
ANSWER_SECTION_DELIMITER = "## Questions & Answers ##"

# 6. Output file name
OUTPUT_FILE = "questions_output.json"


# --- SCRIPT LOGIC ---

def get_repo_parts(url):
    """Extracts owner and repo name from GitHub URL."""
    try:
        parts = url.strip('/').split('/')
        owner, repo = parts[-2], parts[-1]
        return owner, repo
    except IndexError:
        print("❌ Error: Invalid GitHub URL format. Expected 'https://github.com/owner/repo'")
        exit()

def get_md_files_from_repo(owner, repo):
    """Fetches all .md file contents from a GitHub repo."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/"
    files_to_process = []
    
    response = requests.get(api_url)
    if response.status_code != 200:
        print(f"❌ Error fetching repo contents: {response.status_code} - {response.text}")
        return []

    contents = response.json()
    
    for item in contents:
        if item['type'] == 'file' and item['name'].endswith('.md'):
            print(f"📄 Found file: {item['path']}")
            try:
                file_content_response = requests.get(item['url'])
                file_content_response.raise_for_status()
                file_data = file_content_response.json()
                content = base64.b64decode(file_data['content']).decode('utf-8')
                files_to_process.append({'path': item['path'], 'content': content})
            except requests.exceptions.RequestException as e:
                 print(f"⚠️ Warning: Could not fetch content for {item['path']}. Error: {e}")
            except (KeyError, TypeError) as e:
                 print(f"⚠️ Warning: Could not parse content for {item['path']}. Error: {e}")

    print(f"\n✅ Found {len(files_to_process)} Markdown files to process.")
    return files_to_process

def parse_qna_from_content(content, pattern):
    """Splits raw markdown content into a list of Q&A blocks."""
    qna_blocks = re.split(pattern, content)
    return [block.strip() for block in qna_blocks if block.strip()]

def analyze_content_with_llm(qna_block, model):
    """Uses the specified LLM to analyze a single Q&A block and generate JSON."""
    system_prompt = """
    You are an expert technical content analyzer. Your task is to analyze the provided markdown content, which contains a single question and its answer. You must generate a single JSON object with three keys: "type", "difficulty", and "evaluation_rubric".

    1.  **type**: Classify the question into ONE of the following categories: "Theoretical", "Coding", "System Design", "Behavioral", or "Background".
    2.  **difficulty**: Assess the difficulty as "Easy", "Medium", or "Hard".
    3.  **evaluation_rubric**: Create a detailed, objective rubric for evaluating an answer to the question. This rubric should itself be a JSON object where each key is an evaluation criterion (e.g., "Correctness", "Clarity") and the value is another object with descriptions for "Poor", "Good", and "Excellent" performance for that criterion.
    """
    user_prompt = f"Please analyze the following Q&A block and generate the JSON object as instructed.\n\n--- Q&A BLOCK ---\n{qna_block}\n--- END OF BLOCK ---"
    
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"❌ An error occurred with the LLM API call: {e}")
        return None

def main():
    """Main function to run the script."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found. Please create a .env file with your key.")
        return

    owner, repo = get_repo_parts(GITHUB_REPO_URL)
    md_files = get_md_files_from_repo(owner, repo)
    
    if not md_files:
        print("No markdown files found. Exiting.")
        return

    all_questions_data = []

    for file_info in md_files:
        print(f"\nParsing file: {file_info['path']}...")
        
        # --- IMPROVED LOGIC: Isolate the Q&A section first ---
        full_content = file_info['content']
        content_parts = full_content.split(ANSWER_SECTION_DELIMITER)
        
        if len(content_parts) > 1:
            qna_content_block = content_parts[1]
            print(f"   Found '{ANSWER_SECTION_DELIMITER}'. Processing content after it.")
        else:
            # Fallback for files that don't have the specific answer section heading
            qna_content_block = full_content
            print(f"   Warning: Did not find '{ANSWER_SECTION_DELIMITER}'. Processing entire file.")
        
        # Now, parse the isolated block into individual Q&A pairs
        qna_blocks = parse_qna_from_content(qna_content_block, QUESTION_DELIMITER_PATTERN)
        
        if not qna_blocks:
            print(f"⚠️ No Q&A blocks found in the relevant section of {file_info['path']}. Skipping.")
            continue
            
        print(f"   Found {len(qna_blocks)} Q&A pairs in this file.")

        for i, block in enumerate(qna_blocks):
            print(f"   Processing Q&A {i+1}/{len(qna_blocks)} using {MODEL_NAME}...")
            analyzed_data = analyze_content_with_llm(block, MODEL_NAME)
            
            if analyzed_data:
                analyzed_data['source_file'] = file_info['path']
                analyzed_data['question_title'] = block.splitlines()[0].strip().replace('#', '').strip()
                all_questions_data.append(analyzed_data)
                print(f"   ✅ Successfully processed Q&A: \"{analyzed_data['question_title']}\"")
            else:
                print(f"   ⚠️ Failed to process a Q&A block from {file_info['path']}. Skipping.")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_questions_data, f, indent=4)

    print(f"\n✨ All done! Processed {len(all_questions_data)} total Q&A pairs.")
    print(f"Results saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()