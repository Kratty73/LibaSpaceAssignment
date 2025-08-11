import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.vector_stores import VectorStoreQuery, MetadataFilters, ExactMatchFilter
import openai as openai_client
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- 1. Configuration and Initialization ---

# Load environment variables from .env file
load_dotenv()

# Securely get API keys
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not pinecone_api_key or not openai_api_key:
    print("Error: OpenAI or Pinecone API key not found in .env file.")
    exit()

# File paths and model/index settings
SKILL_TAXONOMY_JSON = "skill_taxonomy.json"
PINECONE_INDEX_NAME = "interview-question-bank"
LLM_MODEL = "gpt-4o"

print("Initializing clients and models...")
# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# Initialize OpenAI client
openai_client.api_key = openai_api_key

# Initialize LlamaIndex components
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store)

# Load taxonomy on startup
skill_map = None

# --- 2. FastAPI App Setup ---

app = FastAPI()

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models for request and response bodies
class JobDescriptionRequest(BaseModel):
    job_title: str
    job_description: str

# --- 3. Helper Functions (Adapted from original script) ---

def load_skill_taxonomy(filepath="skill_taxonomy.json"):
    """Loads the taxonomy and creates a reverse map for easy lookups."""
    print("Loading skill taxonomy...")
    try:
        with open(filepath, 'r') as f:
            taxonomy = json.load(f)
        skill_to_category = {skill.lower(): category for category, skills in taxonomy.items() for skill in skills}
        print("Taxonomy loaded successfully.")
        return skill_to_category
    except FileNotFoundError:
        print(f"Error: Taxonomy file not found at '{filepath}'")
        return None

def extract_and_map_skills_from_jd(job_description, skill_map):
    """Uses an LLM to extract raw skills from a JD, then maps them to the taxonomy."""
    print("Analyzing Job Description to extract and map skills...")
    system_prompt = "Your task is to act as a text analysis tool. Scrutinize the following job description and extract only the technical skills, software tools, programming languages, and specific methodologies explicitly mentioned in the text. Do not infer skills from the company's industry (e.g., if the company is in finance, do not add 'financial modeling' unless it's explicitly written). Return the output as a single JSON object with the key 'skills' containing an array of strings."
    
    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": job_description}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        raw_skills = json.loads(content).get("skills", [])
        if not isinstance(raw_skills, list): raw_skills = []

        category_counts = {}
        mapped_skills = []
        for skill in raw_skills:
            skill_lower = skill.lower()
            if skill_lower in skill_map:
                category = skill_map[skill_lower]
                category_counts[category] = category_counts.get(category, 0) + 1
                mapped_skills.append(skill_lower)
        
        print("Skill analysis complete.")
        return {"mapped_skills": mapped_skills, "category_breakdown": category_counts}
    except Exception as e:
        print(f"An error occurred during skill extraction: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract skills from JD.")

def get_available_question_types():
    """
    Gets the unique question types available in the question bank.
    NOTE: In a production system with millions of vectors, this can be slow.
    It's often better to have a separate, pre-computed list of available types.
    For this project, we will use a hardcoded list that matches our data.
    """
    print("Fetching available question types...")
    # This is a placeholder. A real implementation might query Pinecone's metadata.
    return ["Theoretical", "Behavioral", "System Design", "Coding", "Conceptual", "Technical"]


def generate_interview_template(job_title, skill_analysis, available_types):
    """Uses an LLM to generate a structured interview plan."""
    print("Generating dynamic interview template...")
    system_prompt = f"""
    You are an expert Interview Designer for the role of '{job_title}'.
    Based on the skill breakdown from a job description, create a balanced 5-question interview plan.
    For each question, specify a 'question_type' from the available list and a 'search_query' to find a relevant question.
    The search query should be a natural language description of the question's intent.

    Available Question Types: {json.dumps(available_types)}

    Return a single JSON object with one key, "interview_plan", which contains a list of exactly 5 objects.
    Each object must have two keys: "question_type" and "search_query".
    Example:
    {{
        "interview_plan": [
            {{"question_type": "Behavioral", "search_query": "A question about handling conflict with a team member."}},
            {{"question_type": "System Design", "search_query": "A question about designing a scalable data pipeline."}}
        ]
    }}
    """
    user_prompt = f"Skill Breakdown:\n{json.dumps(skill_analysis, indent=2)}"
    
    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        template = json.loads(content)
        print("Interview template generated successfully.")
        return template.get("interview_plan", [])
    except Exception as e:
        print(f"An error occurred during template generation: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate interview template.")

def retrieve_questions(interview_plan):
    """Performs a hybrid search (filter + semantic) for each item in the plan."""
    print("Retrieving questions from Pinecone using hybrid search...")
    retriever = index.as_retriever(similarity_top_k=5) # Retrieve top 5 candidates for de-duplication
    
    final_questions = []
    used_questions = set()

    for item in interview_plan:
        question_type = item.get("question_type")
        search_query = item.get("search_query")

        if not question_type or not search_query:
            continue

        try:
            # LlamaIndex uses MetadataFilters for this kind of query
            query = VectorStoreQuery(
                query_str=search_query,
                filters=MetadataFilters(
                    filters=[ExactMatchFilter(key="question_type", value=question_type)]
                )
            )
            response_nodes = retriever.retrieve(search_query) # Note: LlamaIndex retriever API may vary; adjust if needed.
            
            found_question = False
            for node in response_nodes:
                original_question = node.metadata.get("original_question", "N/A")
                if original_question not in used_questions:
                    final_questions.append(original_question)
                    used_questions.add(original_question)
                    found_question = True
                    break # Move to the next item in the plan
            
            if not found_question:
                 final_questions.append(f"Could not find a unique '{question_type}' question for this topic.")

        except Exception as e:
            print(f"An error occurred while querying for '{search_query}': {e}")
            final_questions.append("Error retrieving question.")
            
    print("Question retrieval complete.")
    return final_questions[:5] # Ensure we only return 5 questions

# --- 4. API Endpoint ---

@app.on_event("startup")
def startup_event():
    """Load the taxonomy when the server starts."""
    global skill_map
    skill_map = load_skill_taxonomy(SKILL_TAXONOMY_JSON)
    if not skill_map:
        print("CRITICAL: Could not load skill taxonomy. The application might not work correctly.")

@app.post("/generate-interview")
def generate_interview_endpoint(request: JobDescriptionRequest):
    """The main endpoint to run the full interview generation pipeline."""
    if not skill_map:
        raise HTTPException(status_code=500, detail="Skill taxonomy is not loaded.")

    # Step 1: Analyze the Job Description
    skill_analysis = extract_and_map_skills_from_jd(request.job_description, skill_map)
    print("\n--- Skill Analysis Results ---")
    print(json.dumps(skill_analysis, indent=2))

    # Step 2: Get available question types
    available_types = get_available_question_types()

    # Step 3: Generate the structured Interview Template
    interview_plan = generate_interview_template(request.job_title, skill_analysis, available_types)
    print("\n--- Generated Interview Plan ---")
    print(json.dumps(interview_plan, indent=2))

    # Step 4: Retrieve the Final Questions using the plan
    final_questions = retrieve_questions(interview_plan)
    print("\n--- Final Interview Package ---")
    print(final_questions)
    
    return {"questions": final_questions}

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
