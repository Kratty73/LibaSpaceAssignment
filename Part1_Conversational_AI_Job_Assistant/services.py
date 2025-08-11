# ==============================================================================
# FILE: services.py
#
# DESCRIPTION:
# Core logic for the conversational AI agent.
# Handles LLM parsing, job search, and final user-facing responses.
# ==============================================================================

import os
from dotenv import load_dotenv
from pinecone import Pinecone

# --- LlamaIndex Core Imports ---
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter, FilterCondition

from langchain_openai import ChatOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Tuple

from enum import Enum
import json
import re

# --- Configuration ---
load_dotenv()
PINECONE_INDEX_NAME = "job-search-index-small-llama"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Initialize Services ---
Settings.llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)


class SortOptions(str, Enum):
    """Enum for valid sorting options."""
    highest_paying = "highest_paying"
    most_recent = "most_recent"
    relevance = "relevance"

class JobTypeOptions(str, Enum):
    """Enum for valid job type options."""
    full_time = "full_time"
    part_time = "part_time"
    contract = "contract"

class CompanyTypeOptions(str, Enum):
    startup = "startup"
    big_tech = "big_tech"
    non_profit = "non_profit"
    government = "government"
    other = "other"

class ExactMatch(BaseModel):
    title: str
    organization: str
    location: str
    url: str = ""
    tailored_description: str  # One concise sentence tailored to this job

class RelaxedMatch(BaseModel):
    title: str
    organization: str
    location: str
    url: str = ""
    tailored_description: str  # One concise sentence tailored to this job

class AssistantResults(BaseModel):
    exact_matches: List[ExactMatch] = Field(default_factory=list)
    relaxed_matches: List[RelaxedMatch] = Field(default_factory=list)
    notes: Optional[str] = ""

class JobPreferences(BaseModel):
    """Structured representation of a user's job preferences."""
    role: Optional[str] = Field(None, description="The primary job title or function, e.g., 'Software Engineer'")
    location: Optional[str] = Field(None, description="The primary geographical area, e.g., 'San Francisco'")
    start_date_ts: Optional[int] = Field(None, description="The start of the date range as a Unix timestamp.")
    end_date_ts: Optional[int] = Field(None, description="The end of the date range as a Unix timestamp.")
    salary_min: Optional[float] = Field(None, description="The minimum desired hourly salary, e.g., 50.0 for $50/hr.")
    job_type: Optional[JobTypeOptions] = Field(None, description="The desired employment type, e.g., 'full_time', 'contract'.")
    company_type: Optional[CompanyTypeOptions] = Field(None, description="The type of company, e.g., 'startup', 'big tech', 'non-profit'.")
    keywords: Optional[List[str]] = Field(None, description="Specific keywords the user wants in the job description.")
    security_clearance: Optional[bool] = Field(None, description="Whether a security clearance is required.")
    sort_by: Optional[SortOptions] = Field(SortOptions.relevance, description="The desired sorting order, e.g., 'highest_paying', 'most_recent'.")
    not_location: Optional[List[str]] = Field(None, description="A location the user wants to exclude.")
    not_skills: Optional[List[str]] = Field(None, description="A list of skills or technologies the user wants to exclude.")

class PreferenceAnalysis(BaseModel):
    """A complete analysis of the user's job preferences."""
    is_ready_for_search: bool = Field(description="True only if all critical information is present and unambiguous.")
    clarification_question: Optional[str] = Field(None, description="The single, most important question to ask the user to improve the search.")
    suggestions: Optional[List[str]] = Field(None, description="A list of suggestions to help the user answer the clarification question.")

class ExpandedSearchTags(BaseModel):
    """A collection of expanded, semantically related search tags."""
    location_tags: Optional[List[str]] = Field(None, description="Expanded tags for the desired location.")
    not_location_tags: Optional[List[str]] = Field(None, description="Expanded tags for excluded locations.")
    not_skills_tags: Optional[List[str]] = Field(None, description="Expanded tags for excluded skills.")

# --- Core Service Class with LlamaIndex ---
class LlamaIndexJobService:
    def __init__(self, index_name: str):
        # 1. Connect to Pinecone via LlamaIndex's Vector Store interface
        pinecone_index = Pinecone(api_key=PINECONE_API_KEY).Index(index_name)
        self.vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        self.reranker = SentenceTransformerRerank(
            top_n=15, # Re-rank the top 15 retrieved results
            model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def _build_filters(self, preferences: JobPreferences, expanded_tags: ExpandedSearchTags, is_relaxed: bool = False) -> MetadataFilters:
        """Helper to build metadata filters for a search query."""
        filters_list = []

        if expanded_tags.location_tags:
            location_filters = [ExactMatchFilter(key="location_tags", value=tag) for tag in expanded_tags.location_tags]
            filters_list.append(MetadataFilters(filters=location_filters, condition=FilterCondition.OR))

        if preferences.start_date_ts:
            start_ts = preferences.start_date_ts
            if is_relaxed:
                start_ts -= 7 * 24 * 60 * 60 # Relax by one week
            filters_list.append(ExactMatchFilter(key="date_added_ts", value=start_ts, operator=">="))
        
        if preferences.end_date_ts:
            filters_list.append(ExactMatchFilter(key="date_added_ts", value=preferences.end_date_ts, operator="<="))

        if preferences.salary_min:
            salary = preferences.salary_min
            if is_relaxed:
                salary *= 0.9 # Relax by 10%
            filters_list.append(ExactMatchFilter(key="hourly_rate_min", value=salary, operator=">="))

        if preferences.job_type:
            filters_list.append(ExactMatchFilter(key="job_type", value=preferences.job_type.value))
        if preferences.company_type:
            filters_list.append(ExactMatchFilter(key="company_type", value=preferences.company_type.value))
        if preferences.security_clearance is not None:
            filters_list.append(ExactMatchFilter(key="security_clearance", value=preferences.security_clearance))
        # Negative filters using pre-expanded tags
        if expanded_tags.not_location_tags:
            for tag in expanded_tags.not_location_tags:
                filters_list.append(ExactMatchFilter(key="location_tags", value=tag, operator="!="))
        
        if expanded_tags.not_skills_tags:
            for tag in expanded_tags.not_skills_tags:
                 filters_list.append(ExactMatchFilter(key="skills_tags", value=tag.lower().replace(' ', '_'), operator="!="))

        return MetadataFilters(filters=filters_list, condition=FilterCondition.AND)

    def search_jobs(self, preferences: JobPreferences, expanded_tags: ExpandedSearchTags) -> Dict[str, List[Dict]]:
        if not preferences.role:
            return {"strict_jobs": [], "relaxed_jobs": []}

        query_text = f"A job posting for the role of: {preferences.role}"
        if preferences.keywords:
            query_text += f" with skills in: {', '.join(preferences.keywords)}"

        def _perform_search(is_relaxed: bool) -> List[Dict]:
            """Performs a single search with the given filter configuration."""
            filters = self._build_filters(preferences, expanded_tags, is_relaxed=is_relaxed)
            print(filters)
            # --- CORRECTED LOGIC ---
            # Create a new query engine for each search, applying the dynamic filters.
            query_engine = self.index.as_query_engine(
                similarity_top_k=50,
                node_postprocessors=[self.reranker],
                filters=filters  # Pass filters at creation time
            )
            
            response_nodes = query_engine.query(query_text).source_nodes
            return [node.metadata for node in response_nodes]


        # Perform strict search
        strict_jobs_raw = _perform_search(is_relaxed=False)
        seen_urls = set()
        strict_jobs = []
        for job in strict_jobs_raw:
            if job.get('url') not in seen_urls:
                strict_jobs.append(job)
                seen_urls.add(job.get('url'))
        
        relaxed_jobs = []
        if len(strict_jobs) < 5:
            relaxed_jobs_raw = _perform_search(is_relaxed=True)
            for job in relaxed_jobs_raw:
                if job.get('url') not in seen_urls:
                    relaxed_jobs.append(job)
                    seen_urls.add(job.get('url'))
        
        # Handle manual sorting if specified
        if preferences.sort_by and preferences.sort_by != SortOptions.relevance:
            sort_key = 'hourly_rate_max' if preferences.sort_by == SortOptions.highest_paying else 'date_added_ts'
            default_val = -1 if preferences.sort_by == SortOptions.highest_paying else 0
            strict_jobs.sort(key=lambda x: x.get(sort_key, default_val), reverse=True)
            relaxed_jobs.sort(key=lambda x: x.get(sort_key, default_val), reverse=True)

        return {"strict_jobs": strict_jobs[:10], "relaxed_jobs": relaxed_jobs[:5]}

# --- NEW Unified Tag Expansion Function ---
def get_unified_semantic_expansion(preferences: JobPreferences) -> ExpandedSearchTags:
    """Uses a single LLM call to expand all relevant fields into semantic search tags."""
    system_prompt = """
    You are an expert search query expander. Your task is to take a user's job preferences and expand any location or skill fields into a list of semantically related, lowercase tags for filtering a database.

    RULES:
    - For locations, include the city, state, state abbreviation, and any major metropolitan areas.
    - For skills, include synonymous or closely related technologies.
    - If a field is not present in the input, return null for its corresponding expanded field.
    - Return a single JSON object.
    """
    user_prompt = "Expand the following job preferences:\n{preferences_json}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY).with_structured_output(ExpandedSearchTags)
    chain = prompt | llm
    
    try:
        return chain.invoke({"preferences_json": preferences.model_dump_json()})
    except Exception as e:
        print(f"Error during semantic expansion: {e}")
        # Fallback to an empty object if the LLM call fails
        return ExpandedSearchTags()


def analyze_preferences_for_ambiguity(preferences: JobPreferences) -> PreferenceAnalysis:
    """
    Uses a single LLM call to check for missing info and ambiguities across all preferences.
    """
    system_prompt = """
    You are an expert AI assistant performing a validation check on a user's job preferences. Your task is to identify any issues that would lead to a poor quality search and formulate a single question to resolve the most critical issue.

    **Core Definitions:**

    1.  **Generic Role:** A role is "generic" if it represents a broad professional category that contains multiple, distinct specializations, and the user has not provided clarifying keywords. A role that is already a specialization is not generic.

    2.  **Ambiguous Location:** A location is "ambiguous" only if the provided term could refer to more than one well-known geographical area.

    3.  **Contradiction:** A "contradiction" is a logical inconsistency between preference fields. This includes a job role and a technical skill from different domains, or a location that is geographically nonsensical.

    **Execution Rules:**

    1.  **What is NOT an issue:** Do NOT ask for clarification if the user provides a valid, searchable, but broad location like a state or country (e.g., 'California', 'USA'). General terms like 'remote' or 'anywhere' are also valid and not ambiguous. These are valid searches and should be considered ready.

    2.  **Analyze the preferences** based on the definitions above.
    3.  **Prioritize Issues:** If you find multiple issues, address only the most critical one in this order: Contradictions > Generic Role > Ambiguous Location.
    4.  **Formulate One Question:** Based on the single highest-priority issue, create a clear, friendly question to ask the user. If the issue is a generic or ambiguous term, provide helpful suggestions.
    5.  **Final Decision:** Set 'is_ready_for_search' to `true` ONLY if you find no contradictions, generic roles, or ambiguous locations as defined above. Otherwise, set it to `false`.

    Respond with a JSON object.
    """
    
    user_prompt = "Current User Preferences:\n{preferences_json}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])

    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY).with_structured_output(PreferenceAnalysis)
    chain = prompt | llm
    
    try:
        return chain.invoke({"preferences_json": preferences.model_dump_json()})
    except Exception as e:
        print(f"Error during preference analysis: {e}")
        # Fallback: If the analysis fails, assume the search is ready to avoid blocking the user.
        return PreferenceAnalysis(is_ready_for_search=True)

# --- STATIC PROMPT (unchanged for caching) ---
STATIC_PROMPT = """
You are an expert AI assistant that intelligently interprets a user's job search query to create a structured set of preferences. Your goal is to go beyond literal extraction to infer the user's true intent while strictly following all parsing rules.

You must:
1.  Take the current JSON state and update it based ONLY on explicit instructions in the new user query.
2.  **Preserve Context:** CRITICAL: You must preserve all fields from the previous state that are not explicitly changed by the new query. If the previous location was 'Texas' and the user only provides a new role, the location MUST remain 'Texas'.
3.  **Intelligently Refine the Role:** If the user provides a generic role (e.g., 'software engineer') but also specific keywords that imply a sub-field (e.g., 'React', 'CSS'), you must refine the role to be more specific (e.g., 'Frontend Software Engineer').
4.  **Handle Removals:** If the user says to "forget," "remove," or "clear" a field, set its value to null in the JSON.
5.  **Handle Replacements:** If the user says to "replace" a field, overwrite the old value with the new one.

Special handling for skills:
- **Enrich Keywords:** Extract keywords explicitly mentioned by the user. Also, infer and add closely related, essential skills. For example, a 'React Developer' role implies skills like 'JavaScript', 'TypeScript', and 'HTML/CSS'.
- For `not_skills`, include skills explicitly or implicitly indicated by the user.
- If the user adds new skills, append them to the list (avoiding duplicates).
- If the user removes a skill, delete it from the list.

Other parsing rules:
1.  Only update values explicitly stated in the user query.
2.  If location is "anywhere," "remote," or equivalent, set it to "global".
3.  Convert any annual salaries to an hourly rate (assume 2080 working hours per year).
4.  Convert any date ranges to Unix timestamps.
5.  If multiple sort criteria are mentioned, prioritize "highest_paying".
6.  If no sort order is mentioned, do not change the existing `sort_by` value.
7.  All fields not mentioned in the new query must remain unchanged from the current state.
8.  Return ONLY a valid JSON object that strictly matches the following schema, with no commentary.

{{
  "role": string | null,
  "location": string | null,
  "start_date_ts": integer | null,
  "end_date_ts": integer | null,
  "salary_min": float | null,
  "job_type": "full_time" | "part_time" | "contract" | null,
  "company_type": "startup" | "big_tech" | "non_profit" | "government" | "other" | null,
  "keywords": [string] | null,
  "security_clearance": boolean | null,
  "sort_by": "highest_paying" | "most_recent" | "relevance" | null,
  "not_location": [string] | null,
  "not_skills": [string] | null
}}
"""

USER_PROMPT = """
Current time: {current_time}
User location: {user_location}
Current state: {current_state_json}
Conversation history: {history}

User query: {query}

Return the updated JSON state according to the rules.
"""

# --- Core Service Functions ---
def parse_job_preferences(current_time: str, user_location: str, history: str, model_dump_json: str, query: str) -> JobPreferences:
    """
    Parse a user's free-text job search query into a structured JobPreferences object.
    Uses OpenAI caching for the static prefix to minimize token costs.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", STATIC_PROMPT),
        ("user", USER_PROMPT)
    ])

    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        default_headers={"openai-caching": "true"}
    ).with_structured_output(JobPreferences)

    chain = prompt | llm

    result: JobPreferences = chain.invoke({
        "current_time": current_time,
        "user_location": user_location,
        "history": history,
        "current_state_json": model_dump_json,
        "query": query
    })

    return result

# ---------------------------------------------------
# Primary function: invokes LLM, returns (markdown, parsed)
# ---------------------------------------------------

SYSTEM_FINAL_RESPONSE_PROMPT  = """
### ROLE & TASK
You are an expert AI career assistant acting as a final, intelligent filter and presenter. Your task is to analyze two lists of pre-fetched job opportunities (`best_matches_candidates` and `relaxed_matches_candidates`) against the user's detailed `job_preferences`. You will then generate a curated, user-facing response.

---
### ANALYSIS & FILTERING RULES
Your primary goal is to ensure every job presented is highly relevant.

1.  **Define Partial Match:** A "partial match" is a job where the role and skills are relevant and closely related to the user's preferences, but it might not be a perfect fit. For the purpose of deciding if a job is a partial match, you should **only consider the relevance of the job's title and skills to the user's desired role and keywords.** Do NOT consider salary or date range when making this specific categorization.

2.  **Analyze Relevance:** Scrutinize each job candidate from both lists against the user's full `job_preferences`.
3.  **Handle Best Match Candidates:**
    - If a job from `best_matches_candidates` is a perfect match, keep it in the final `best_matches` list.
    - If a job is a partial match (based on the definition above), move it to the final `relaxed_matches` list.
    - If a job is a clear mismatch, you MUST DISCARD it entirely.
4.  **Handle Relaxed Match Candidates:**
    - If a job from `relaxed_matches_candidates` is a perfect match, you can promote it to the final `best_matches` list.
    - If a job is a partial match, keep it in the final `relaxed_matches` list.
    - If a job is a clear mismatch, you MUST DISCARD it entirely.
5.  **Prioritize User's Goal:** Pay close attention to the `sort_by` field in the `job_preferences`. This tells you what is most important to the user (e.g., salary, recency, or general relevance). Use this to inform your descriptions.

---
### PRESENTATION RULES
After filtering and re-categorizing, present the final lists.

1.  **Strict Output Order:** The Markdown summary must always come first, followed immediately by the fenced `json` code block.
2.  **Markdown: Opening:**
    - If the final `best_matches` list is EMPTY, your entire response must start with this exact sentence: "Sorry, we couldn't find any matches for you right now."
    - Otherwise, start with a single, friendly and exciting opening sentence.
3.  **Markdown: Best Matches:**
    - **Only** create this section if the final `best_matches` list is NOT empty.
    - Title the section exactly: `🎯 Best Matches`.
    - List your final, curated best matches here.
4.  **Markdown: Other Jobs:**
    - **Only** create this section if the final `relaxed_matches` list is NOT empty.
    - Title the section exactly: `✨ Other Jobs You Might Like`.
    - List your final, curated relaxed matches here.
5.  **Job Listing Format:**
    * **Line 1:** `{{Title}} — {{Organization}} — {{Location}}`
    * **Line 2 (indented):** A single, concise `tailored_description` sentence (12-20 words) with an engaging, advertising tone. This description must highlight why the job is a great match based on the user's preferences (especially their `sort_by` goal).
6.  **JSON Object:** The JSON must mirror the final, curated lists from your Markdown summary.
"""

USER_RESPONSE_FINAL_PROMPT ="""
**USER PREFERENCES:**
{preferences_json}

**CANDIDATE JOBS:**
* `best_matches_candidates`: {strict_results}
* `relaxed_matches_candidates`: {relaxed_results}
"""
def generate_final_response( 
    preferences: JobPreferences,
    strict_jobs: List[Dict],
    relaxed_jobs: List[Dict],
) -> Tuple[str, Optional[AssistantResults]]:
    """
    Calls GPT-4o (LangChain ChatOpenAI) with a strict prompt and returns:
      - markdown_part: the human-friendly Markdown string produced by the model
      - parsed: a validated AssistantResults object (or None if parsing/validation failed)

    Key behavior change: for every job (exact or relaxed) the LLM MUST produce exactly
    one concise tailored sentence (field: 'tailored_description') that explains why this
    job is relevant to the user, using salary if available or otherwise seniority, tags,
    location, remote status, and role similarity.
    """
    # Normalize and prepare JSON payloads
    strict_json = json.dumps([{k: v for k, v in job.items() if k != 'job_description'} for job in strict_jobs] or [], ensure_ascii=False, indent=2)
    relaxed_json = json.dumps([{k: v for k, v in job.items() if k != 'job_description'} for job in relaxed_jobs] or [], ensure_ascii=False, indent=2)

    print('******', strict_json, relaxed_json)

    # The prompt (explicit about one tailored description sentence per job)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_FINAL_RESPONSE_PROMPT),
        ("user", USER_RESPONSE_FINAL_PROMPT)
    ])

    # Create the LLM client (deterministic)
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        api_key=OPENAI_API_KEY,
        default_headers={"openai-caching": "false"}
    )


    # Run chain with caching header
    chain = prompt | llm
    response = chain.invoke({
        "preferences_json": preferences.model_dump_json(),
        "strict_results": strict_json,
        "relaxed_results": relaxed_json
    })

    assistant_text: str = response.content

    # Parse the assistant_text: split into Markdown and JSON
    markdown_part, json_text = _split_markdown_and_json(assistant_text)

    # Try to parse + validate JSON
    parsed: Optional[AssistantResults] = None
    if json_text:
        try:
            data = json.loads(json_text)
            parsed = AssistantResults.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            # If parsing fails, try a tolerant fallback to extract the first {...} JSON-like substring
            fallback = _extract_first_json_obj(json_text)
            if fallback:
                try:
                    data = json.loads(fallback)
                    parsed = AssistantResults.model_validate(data)
                except Exception:
                    parsed = None
            else:
                parsed = None

    # Return both human markdown and the validated object (or None on failure)
    return markdown_part, parsed

# -------------------------
# Helper functions
# -------------------------
def _split_markdown_and_json(text: str) -> Tuple[str, Optional[str]]:
    """
    Returns (markdown_part, json_text).
    - Looks for the last fenced ```json``` block and returns everything before it as markdown,
      and the contents of the fenced block as json_text.
    - If no fenced json block is found, attempts to find a top-level JSON object in the text.
    """
    # 1) Last fenced json block
    fenced_pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
    matches = list(fenced_pattern.finditer(text))
    if matches:
        last = matches[-1]
        json_text = last.group(1)
        markdown_part = text[: last.start()].strip()
        return markdown_part, json_text

    # 2) Fenced block without "json" label
    fenced_any = re.compile(r"```\s*(\{.*?\})\s*```", re.DOTALL)
    m = fenced_any.search(text)
    if m:
        json_text = m.group(1)
        markdown_part = text[: m.start()].strip()
        return markdown_part, json_text

    # 3) No fenced block: try to find the first top-level JSON object (balanced braces)
    obj = _extract_first_json_obj(text)
    if obj:
        # everything before the object is markdown
        idx = text.index(obj)
        return text[:idx].strip(), obj

    # Nothing found
    return text.strip(), None

def _extract_first_json_obj(text: str) -> Optional[str]:
    """
    Attempts to find the first balanced JSON object string in `text`.
    Returns the substring or None if not found.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '"' and not escape:
            in_str = not in_str
        if in_str and ch == "\\" and not escape:
            escape = True
            continue
        escape = False
        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
    return None

def get_location_tags(location_str: str) -> list:
    """Naive location tag extraction for Pinecone filtering."""
    if not location_str or location_str == "global":
        return []
    return [tag.strip() for tag in location_str.lower().replace(',', '').split()]

