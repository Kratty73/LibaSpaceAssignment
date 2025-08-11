import requests
from bs4 import BeautifulSoup
import json
import time
import random
import os

# --- Configuration ---
# We are targeting well-structured public pages and repositories.
# This is more reliable, faster, and respectful than hitting dynamic sites directly.
BASE_URLS = {
    # 'behavioral': 'https://www.techinterviewhandbook.org/behavioral-questions/',
    # 'algorithms': 'https://www.techinterviewhandbook.org/algorithms/algorithms-introduction/',
    # 'system_design': 'https://www.techinterviewhandbook.org/system-design/',
    # Added a new source for high-quality LeetCode problems
    'leetcode_patterns': 'https://seanprashad.com/leetcode-patterns/'
}
OUTPUT_FILE = 'leetcode_interview_questions.json'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def fetch_page_content(url):
    """
    Fetches the HTML content of a given URL.
    Includes error handling and a polite delay.
    """
    try:
        print(f"Fetching content from: {url}")
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        # Polite delay to avoid overwhelming the server
        time.sleep(random.uniform(1, 3))
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def parse_behavioral_questions(html_content):
    """
    Parses behavioral questions from the Tech Interview Handbook.
    """
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, 'html.parser')
    questions = []
    content_div = soup.find('article')
    if not content_div:
        return []

    for li in content_div.find_all('li'):
        question_text = li.get_text(strip=True)
        if '?' in question_text and len(question_text) > 20:
             questions.append({
                "category": "Behavioral",
                "question_text": question_text,
                "source_url": BASE_URLS['behavioral']
            })
    return questions

def parse_conceptual_questions(html_content, category):
    """
    Parses conceptual questions (Algorithms, System Design) from topic headers.
    """
    if not html_content:
        return []
    
    soup = BeautifulSoup(html_content, 'html.parser')
    questions = []
    content_div = soup.find('article')
    if not content_div:
        return []

    for header in content_div.find_all(['h2', 'h3']):
        question_text = header.get_text(strip=True)
        if question_text and len(question_text) > 5:
            conceptual_question = f"Can you explain the concept of '{question_text}' and its importance in {category}?"
            questions.append({
                "category": category,
                "question_text": conceptual_question,
                "source_url": BASE_URLS.get(category.lower().replace(" ", "_"))
            })
    return questions

def parse_leetcode_questions(html_content):
    """
    Parses LeetCode questions from the seanprashad.com/leetcode-patterns page.
    This page has a very clean table structure.
    """
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, 'html.parser')
    questions = []
    
    # The questions are in tables. We find all tables on the page.
    tables = soup.find_all('table')
    for table in tables:
        # Get the pattern name from the heading preceding the table
        pattern_header = table.find_previous_sibling('h2')
        pattern = pattern_header.get_text(strip=True) if pattern_header else "General"

        # Find all rows in the table body
        for row in table.find('tbody').find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 3:
                # Extract data from cells
                question_cell = cells[1]
                difficulty_cell = cells[2]

                link_tag = question_cell.find('a')
                if not link_tag:
                    continue
                
                question_title = link_tag.get_text(strip=True)
                question_url = link_tag['href']
                difficulty = difficulty_cell.get_text(strip=True)

                questions.append({
                    "category": "Coding Challenge",
                    "question_text": f"LeetCode ({pattern}): {question_title}",
                    "difficulty": difficulty,
                    "source_url": question_url,
                    "tags": ["LeetCode", pattern]
                })
    return questions

def main():
    """
    Main function to orchestrate the crawling process.
    """
    all_questions = []
    print("Starting crawler...")

    # --- Crawl Behavioral Questions ---
    # behavioral_html = fetch_page_content(BASE_URLS['behavioral'])
    # if behavioral_html:
    #     behavioral_questions = parse_behavioral_questions(behavioral_html)
    #     all_questions.extend(behavioral_questions)
    #     print(f"-> Found {len(behavioral_questions)} behavioral questions.")

    # # --- Crawl Algorithms & System Design Topics ---
    # for category_name, url in [('Algorithms', BASE_URLS['algorithms']), ('System Design', BASE_URLS['system_design'])]:
    #     html_content = fetch_page_content(url)
    #     if html_content:
    #         conceptual_questions = parse_conceptual_questions(html_content, category_name)
    #         all_questions.extend(conceptual_questions)
    #         print(f"-> Found {len(conceptual_questions)} {category_name} conceptual questions.")
            
    # --- Crawl LeetCode Coding Challenges ---
    leetcode_html = fetch_page_content(BASE_URLS['leetcode_patterns'])
    if leetcode_html:
        leetcode_questions = parse_leetcode_questions(leetcode_html)
        all_questions.extend(leetcode_questions)
        print(f"-> Found {len(leetcode_questions)} LeetCode coding challenges.")

    # --- Save the results to a file ---
    if all_questions:
        print(f"\nTotal questions crawled: {len(all_questions)}")
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(all_questions, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved questions to '{os.path.abspath(OUTPUT_FILE)}'")
        except IOError as e:
            print(f"Error writing to file {OUTPUT_FILE}: {e}")
    else:
        print("No questions were crawled. Check for errors above.")

if __name__ == '__main__':
    main()
