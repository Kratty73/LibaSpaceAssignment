# Part 1: Conversational AI Job Assistant

This project is a Conversational AI Job Assistant built with FastAPI, LlamaIndex, and Pinecone. It provides a natural language interface for job seekers to find relevant positions by describing their preferences conversationally. The assistant can understand user intent, ask clarifying questions, and query a job database to return tailored recommendations.

---
## How This Project Meets the Requirements

This implementation directly addresses each of the core tasks outlined in the assignment brief:

* **✅ Natural Language Preferences:** The `/chat` endpoint accepts free-text user queries (e.g., 'Looking for a Data Analyst role in the Bay Area at a startup').

* **✅ Attribute Parsing:** The `services.parse_job_preferences` function uses a structured output call to **GPT-4o** to parse attributes like **role, location, salary**, and more into a validated Pydantic model.

* **✅ Clarifying Questions:** The `services.analyze_preferences_for_ambiguity` function explicitly checks for vague or contradictory user intent and uses an LLM to generate a relevant follow-up question, preventing low-quality searches.

* **✅ Query Job API:** The `LlamaIndexJobService.search_jobs` method queries a Pinecone vector index (acting as our mock job API) using a hybrid search strategy to find the most relevant job matches.

* **✅ Memory/Session Tracking:** The `main.py` file uses a simple in-memory dictionary (`conversation_preferences`) to store the state of each user's job search, allowing for multi-turn conversational context.

* **✅ Prompt Design & Error Handling:** The `services.py` file contains detailed, role-based system prompts designed for robustness. Error handling is implemented for API failures, invalid user inputs (e.g., future dates), and safety via OpenAI's moderation endpoint.

---
## Tech Stack Rationale

The tech stack was chosen to create a modern, scalable, and efficient RAG pipeline:

* **FastAPI:** Chosen for its high performance, asynchronous capabilities, and automatic documentation, making it ideal for building a production-ready API layer.
* **LlamaIndex:** Used as the primary data framework to connect to and query the Pinecone vector store. Its high-level abstractions simplify the process of building and executing complex search queries.
* **Pinecone:** Selected as the vector database for its scalability and speed in performing semantic searches over a large corpus of job listings.
* **GPT-4o:** Used as the core reasoning engine for its advanced instruction-following, structured data generation (JSON mode), and nuanced understanding of user intent, which is critical for both parsing and response generation.

---
## Prompt Design and Error Handling

The reliability of this system hinges on its prompt engineering. The key prompts in `services.py` are designed for specific, chained tasks:

1.  **Preference Parsing Prompt (`STATIC_PROMPT`):** This is the most complex prompt. It instructs the LLM to act as an intelligent entity extractor.
    * **Key Feature:** It is designed to be **stateful**. It takes the previous state of the user's preferences and the new query, and returns an updated state. This is crucial for multi-turn conversations, as it prevents the model from forgetting previously stated preferences.
    * **Error Handling:** It includes rules for handling removals ("forget my location") and replacements, making the conversation more natural and robust.

2.  **Ambiguity Analysis Prompt:** This prompt acts as a "validation gate" before any search is performed.
    * **Key Feature:** It is given strict definitions of what constitutes an "ambiguous" or "generic" query. This prevents the assistant from being overly pedantic (e.g., it won't ask for clarification on a broad but valid location like "California"). It is instructed to formulate a single, high-impact question to resolve the most critical issue.

3.  **Final Response Prompt (`SYSTEM_FINAL_RESPONSE_PROMPT`):** This prompt synthesizes the search results into a user-facing response.
    * **Key Feature:** It instructs the model to generate a `tailored_description` for each job match. This forces the LLM to provide a concise, value-driven reason for *why* a particular job is a good fit, directly fulfilling a core requirement of the assignment.

---
## Performance and Future Improvements

While the current implementation is a robust prototype, several key areas could be improved to make it production-ready.

### Performance Analysis (The "Slow" Factor)

The primary bottleneck in the current system is **latency**, which is caused by the multiple, sequential LLM calls required for each turn of the conversation (Parse -> Analyze -> Search -> Generate Response). A single user query can take several seconds to process.

### Actionable Improvements

1.  **Implement a Caching Layer:**
    * **What:** Introduce a high-speed, in-memory cache like **Redis**.
    * **Why:** Many user queries are repetitive. By caching the results of LLM calls (especially the final generated response for a given set of preferences), we can serve subsequent identical requests instantly, dramatically reducing both latency and API costs.

2.  **Optimize LLM Calls:**
    * **What:** Transition from `gpt-4o` to a faster, cheaper model like `gpt-4o-mini` for simpler tasks like the initial preference parsing.
    * **Why:** Not every step requires the full power of GPT-4o. Using a smaller model for less complex reasoning tasks can cut latency by half or more for that specific step without a significant drop in quality.

3.  **Enhance State Management and Logic:**
    * **What:** Incorporate a more sophisticated state machine. This would involve adding support for more complex user logic (e.g., handling 'OR' conditions like 'roles in New York OR remote'), better geospatial awareness for locations, and potentially storing older states to understand a user's evolving search criteria over time.
    * **Why:** This would move the assistant from a simple preference store to a more intelligent agent that can handle nuanced, multi-faceted user requests and better understand the user's journey, leading to more accurate and satisfying job matches.

4.  **Persistent Session Management:**
    * **What:** Replace the in-memory `conversation_preferences` dictionary with a persistent store like **Redis** or a database.
    * **Why:** The current in-memory approach means all user history is lost if the server restarts. A persistent store is essential for a stateful, production-grade application, allowing for long-term user memory and a more robust user experience.

5.  **Bugs:**
    * **What:** The Presenting LLM call is currently failing to accurately filter relevant jobs, while the Ambiguity Flagging LLM is marking reasonable cases as ambiguous, making the system appear less intelligent.
