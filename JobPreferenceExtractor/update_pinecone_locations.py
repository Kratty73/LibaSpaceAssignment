# ==============================================================================
# FILE: update_pinecone_locations.py
#
# DESCRIPTION:
# This script fetches existing records from a Pinecone index, uses an LLM
# to extract standardized location_tags from the location_literal field,
# and then updates the records in Pinecone with this new metadata.
# This avoids a costly full re-ingestion of data.
# ==============================================================================

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import List, Dict
from tqdm import tqdm

# --- LlamaIndex & AI Imports ---
from llama_index.llms.openai import OpenAI
from llama_index.core.program import LLMTextCompletionProgram
from pydantic import BaseModel, Field

# --- Configuration ---
load_dotenv()
PINECONE_INDEX_NAME = "job-search-index-small-llama"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BATCH_SIZE = 100 # Process 100 records at a time

# --- Pydantic Model for Location Tag Extraction ---
class LocationTags(BaseModel):
    """Structured representation of geographical tags."""
    location_tags: List[str] = Field(
        description="A list of standardized, lowercase geographical tags (e.g., city, state, state abbreviation)."
    )

def update_pinecone_with_location_tags():
    """
    Fetches records from Pinecone, extracts location tags, and updates them.
    """
    # 1. Initialize services
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

    print(f"Connected to Pinecone index '{PINECONE_INDEX_NAME}'.")
    index_stats = index.describe_index_stats()
    total_vectors = index_stats.total_vector_count
    print(f"Found {total_vectors} total vectors to process.")

    if total_vectors == 0:
        print("Index is empty. Nothing to update.")
        return

    # 2. Define the LLM program for extracting location tags
    extraction_prompt_template = """
    From the provided location string, extract a list of standardized, lowercase geographical tags.
    Include the city, state, and state abbreviation if available.
    Also, include any common nicknames or alternative names for the location (e.g., for 'San Francisco', include 'sf' and 'sf bay area').

    Location: "{location_literal}"
    """
    program = LLMTextCompletionProgram.from_defaults(
        output_cls=LocationTags,
        prompt_template_str=extraction_prompt_template,
        llm=llm,
    )

    # 3. Fetch all vector IDs from the index
    # Note: For very large indexes (>1M vectors), a paginated approach would be better.
    # This approach is suitable for up to ~1M vectors.
    print("Fetching all vector IDs from the index...")
    all_ids = [v.id for v in index.query(vector=[0]*1536, top_k=10000, include_metadata=False).matches]
    print(f"Fetched {len(all_ids)} IDs.")

    # 4. Process records in batches
    for i in tqdm(range(0, len(all_ids), BATCH_SIZE), desc="Processing Batches"):
        batch_ids = all_ids[i:i + BATCH_SIZE]
        
        # Fetch the full records for the current batch
        fetch_response = index.fetch(ids=batch_ids)

        # This list will hold records that need updating in this batch
        records_to_update_in_batch = []

        for vec_id, vector_data in fetch_response.vectors.items():
            metadata = vector_data.metadata
            location_literal = metadata.get("location_literal")

            if not location_literal or "location_tags" in metadata:
                continue

            try:
                extracted_data: LocationTags = program(location_literal=location_literal)
                
                # Add the record to our list to be updated
                records_to_update_in_batch.append({
                    "id": vec_id,
                    "set_metadata": {
                        "location_tags": extracted_data.location_tags
                    }
                })

            except Exception as e:
                print(f"Skipping vector {vec_id} due to extraction error: {e}")
                continue
        
        # Update the records in Pinecone if there's anything to update
        if records_to_update_in_batch:
            try:
                # --- CORRECTED LOGIC ---
                # Loop through the list and update each record individually.
                for record in records_to_update_in_batch:
                    index.update(
                        id=record["id"], 
                        set_metadata=record["set_metadata"]
                    )
                tqdm.write(f"Successfully updated {len(records_to_update_in_batch)} records in batch {i//BATCH_SIZE + 1}.")
            except Exception as e:
                tqdm.write(f"Error updating batch {i//BATCH_SIZE + 1}: {e}")

    print("\nUpdate process complete.")

if __name__ == "__main__":
    update_pinecone_with_location_tags()
