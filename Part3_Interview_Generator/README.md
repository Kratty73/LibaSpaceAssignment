# Part 3: Job Description-Based Interview Generator

This project is an end-to-end system that converts a raw job description (JD) into a structured, role-specific interview package. It leverages a custom-built skill taxonomy and a vectorized question bank to ensure the generated questions are relevant, balanced, and tailored to the specific requirements of the role.

---

## ✅ How This Project Meets the Requirements

This implementation directly addresses each of the core tasks outlined in the assignment brief:

* **Extract Structured Info & Build Taxonomy**: The pipeline begins by using a powerful LLM (`gpt-4o-mini`) to extract a flat list of over 11,000 unique skills from a corpus of 2,250 real-world job descriptions. A second script then classifies each unique skill into one of several high-level categories, creating a robust, data-driven two-level skill taxonomy.
* **Collect Real-World Interview Questions**: The system is designed to work with a question bank populated from real-world sources. For this demo, we use a JSON file of pre-collected questions.
* **Generate Tailored Questions**: The system generates a balanced set of questions by first creating a dynamic interview plan. This plan specifies the `question_type` (e.g., Behavioral, Coding, System Design) and a semantic `search_query` for each question, ensuring a comprehensive interview.
* **Include Rich Metadata**: The data ingestion script (`pinecone_ingestion_code`) enriches each question with critical metadata, including its `question_type` and `difficulty`, before storing it in the vector database.
* **Match with Question Bank**: The core of the application uses a hybrid search strategy. It first uses metadata filters to narrow the search to the correct `question_type` and then uses a semantic vector search to find the most contextually relevant question within that subset.
* **Output a Custom Interview Package**: The final output of the API is a clean JSON object containing a list of 5 tailored interview questions ready for the user.
* **Optional Demo**: A complete demo is provided, featuring a FastAPI backend that serves the generation logic and a standalone HTML/JavaScript front-end that provides an interactive, agent-style interview experience.

---

## 🛠️ Tech Stack Rationale

The tech stack was chosen to create a modern, scalable, and intelligent pipeline:

* **FastAPI**: Used to build the high-performance, asynchronous API that serves the interview generation logic.
* **LlamaIndex**: Serves as the primary data framework for interacting with the Pinecone vector store. Its abstractions for vector search and metadata filtering are crucial for the hybrid search implementation.
* **Pinecone**: Chosen as the vector database for its scalability and efficiency in handling high-dimensional vector embeddings and performing fast, filtered semantic searches.
* **GPT-4o / gpt-4o-mini**: Used for all reasoning and generation tasks. `gpt-4o-mini` is used for the bulk data processing (skill extraction and classification) due to its cost-effectiveness, while the more powerful `gpt-4o` is used in the real-time pipeline for the nuanced tasks of analyzing JDs and generating dynamic interview plans.

---

## ⚙️ Architecture and Pipeline

The application follows a multi-step pipeline orchestrated by the FastAPI server:

1.  **Skill Analysis**: When a JD is received, the system first extracts a list of raw skills. It then uses the pre-built `skill_taxonomy.json` to map these skills to their respective categories, creating a profile of the job's requirements (e.g., 60% Data Engineering, 40% Cloud & DevOps).
2.  **Dynamic Template Generation**: This skill profile is then passed to `GPT-4o`, which acts as an "Interview Designer." It generates a structured JSON plan, defining the `question_type` and a semantic `search_query` for each of the 5 questions in the interview.
3.  **Hybrid Search Retrieval**: The system iterates through the generated plan. For each item, it executes a hybrid search against the Pinecone index using LlamaIndex. It first applies a strict metadata filter for the `question_type` and then performs a semantic vector search using the `search_query`.
4.  **De-duplication and Assembly**: The system retrieves the top candidate for each search, ensures the same question is not used twice, and assembles the final list of 5 questions.

---

## 💡 Key Design Decisions

* **Two-Level Taxonomy via Classification**: Instead of attempting to generate a complex, multi-level graph (which is unreliable with current LLMs), I opted for a more robust, two-level taxonomy. This was created by first extracting a flat list of skills from data and then using an LLM for a simple and reliable classification task. This provides a clean, predictable structure that is ideal for this application.
* **Dynamic Template Generation**: The system does not rely on static, hardcoded interview templates. By using an LLM to generate a plan on the fly, it can adapt to any job title and skill combination, making it highly scalable and flexible.
* **Hybrid Search for Precision**: Relying on pure semantic search alone cannot guarantee a balanced interview. By combining a semantic search for relevance with strict metadata filtering for question types, the system ensures the final package is both contextually appropriate and structurally sound.

---

## 🚀 Future Improvements

* **Advanced De-duplication**: The current de-duplication logic prevents the exact same question from appearing twice. A more advanced implementation would use vector similarity to ensure that the retrieved questions are not just textually unique but also conceptually distinct.
* **Feedback Loop for Question Quality**: A mechanism could be introduced for users to rate the quality and relevance of the generated questions. This feedback could be used to fine-tune the embedding model or adjust the ranking of questions in the database over time.
* **Caching Layer**: To improve performance and reduce API costs, a caching layer (e.g., Redis) could be implemented to store the results of the full pipeline. If the same JD is submitted twice, the cached interview package could be served instantly.
* **Integration with Part 1**: The skills extracted and analyzed in this part could be used to enrich the job data in Part 1, providing more accurate tags for job seekers and improving the quality of job recommendations.
* **Dynamic Graph-Based Taxonomy**: The current two-level taxonomy is robust for this application. A more advanced implementation could leverage an LLM's knowledge to build a complete, connected skill graph (an ontology). This would allow the system to understand more nuanced relationships (e.g., 'PyTorch' is a framework for 'Deep Learning', which is a sub-field of 'Machine Learning'), enabling even more precise skill matching and interview generation.
