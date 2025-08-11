# ==============================================================================
# FILE: main.py
#
# DESCRIPTION:
# This is the main FastAPI application file. It sets up the API server,
# defines the endpoints, and manages the conversation state and control flow.
#
# INSTRUCTIONS:
# 1. Run the server with: `uvicorn main:app --reload`
# ==============================================================================

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict
from openai import OpenAI
import os
from dotenv import load_dotenv
# Import the core logic from our services file
import services
from datetime import datetime
import time
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the LlamaIndex service once when the app starts
    app.state.job_service = services.LlamaIndexJobService(services.PINECONE_INDEX_NAME)
    yield
    # Clean up resources if needed

# --- API Setup ---
app = FastAPI(
    title="AI Job Assistant API",
    description="A conversational API to help users find jobs.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Data Models for API Requests/Responses ---
class ChatRequest(BaseModel):
    user_id: str
    query: str
    history: List[Dict[str, str]]

class ChatResponse(BaseModel):
    response: str
    history: List[Dict[str, str]]

# In-memory store for conversation state (for this demo)
conversation_preferences: Dict[str, services.JobPreferences] = {}
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest, request: Request):
    """Main endpoint for handling user conversation."""
    user_id = chat_request.user_id
    query = chat_request.query
    history = chat_request.history

    # 1. Moderation Check (Safety First)
    mod_response = openai_client.moderations.create(input=query)
    if mod_response.results[0].flagged:
        response_text = "I'm sorry, I can't assist with that request. Let's keep our conversation professional and focused on job searching."
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response_text})
        return ChatResponse(response=response_text, history=history)

    # 2. Parse the current query to understand user intent
    current_time = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p %Z")
    stored_prefs = conversation_preferences.get(user_id, services.JobPreferences())
    current_prefs = services.parse_job_preferences(current_time, 'Chapel Hill, NC', history, stored_prefs.model_dump_json(exclude_none=False), query)

    # 3. Update and manage conversation state
    new_info = current_prefs.model_dump(exclude_unset=True)
    updated_data = {**stored_prefs.model_dump(), **new_info}
    final_prefs = services.JobPreferences(**updated_data)
    conversation_preferences[user_id] = final_prefs
    print("Updated Preferences:", final_prefs.model_dump_json(indent=2))

    # 4. Decide whether to search or ask for more info
    if not final_prefs.role or not final_prefs.location:
        if not final_prefs.role:
            response_text = "That's a good start! What kind of role are you looking for?"
        else:
            response_text = f"Great, a {final_prefs.role} role. Where are you interested in working?"
    else:
        # 4. Validate Inputs and Handle Edge Cases
        # Fictional/Unrecognized Location Check
        # if final_prefs.location and final_prefs.location != "global":
        #     if len(final_prefs.location.split()) > 1 and not services.get_location_tags(final_prefs.location):
        #         response_text = f"I couldn't recognize '{final_prefs.location}' as a real-world place. Could you please provide a valid city or country?"
        #         history.append({"role": "user", "content": query})
        #         history.append({"role": "assistant", "content": response_text})
        #         return ChatResponse(response=response_text, history=history)

        # Date Range Check

        if final_prefs.start_date_ts:
            current_ts = int(time.time())
            one_year_ago_ts = current_ts - (365 * 24 * 60 * 60)
            if final_prefs.start_date_ts > current_ts:
                response_text = "It looks like you've asked for a date in the future. I can only search for jobs that have already been posted. Please provide a date in the past."
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": response_text})
                return ChatResponse(response=response_text, history=history)
            if final_prefs.start_date_ts < one_year_ago_ts:
                response_text = "I can only search for jobs posted within the last year. Please provide a more recent date."
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": response_text})
                return ChatResponse(response=response_text, history=history)
        
        analysis = services.analyze_preferences_for_ambiguity(final_prefs)

        if not analysis.is_ready_for_search:
            # If the analysis says we need more info, use its generated question.
            response_text = analysis.clarification_question
            
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response_text})
            return ChatResponse(response=response_text, history=history)
        
        # 5. Perform the search and hydrate results
        expanded_tags = services.get_unified_semantic_expansion(final_prefs)
        jobs_dict = request.app.state.job_service.search_jobs(final_prefs, expanded_tags)

        print(jobs_dict)

        strict_jobs = jobs_dict.get('strict_jobs', [])
        relaxed_jobs = jobs_dict.get('relaxed_jobs', [])

        markdown_part, _ = services.generate_final_response(final_prefs, strict_jobs, relaxed_jobs)
        
        # 6. Generate the final, reasoned response
        # markdown_part, _ = services.generate_final_response(query, strict_jobs, relaxed_jobs)
        response_text = markdown_part or ""
    
    # 7. Update history and return response
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": response_text})

    return ChatResponse(response=response_text, history=history)

@app.get("/")
def read_root():
    return {"message": "AI Job Assistant API is running. Use the /chat endpoint to interact."}
