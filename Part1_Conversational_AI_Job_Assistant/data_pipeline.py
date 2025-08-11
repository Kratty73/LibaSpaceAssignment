# ==============================================================================
# FILE: data_pipeline.py (Refactored for Small Model & Enriched Text)
#
# DESCRIPTION:
# This script uses a two-stage process with LlamaIndex:
# 1. Pre-processing: For each job, it uses an LLM to extract rich metadata.
# 2. Ingestion: It creates an enriched "golden document" by combining the
#    original text with the extracted metadata, then uses an IngestionPipeline
#    to chunk, embed (with text-embedding-3-small), and store in Pinecone.
# ==============================================================================

import os
import pandas as pd
from dotenv import load_dotenv
from typing import List, Optional
import math
import random
from datetime import datetime, timedelta

# --- LlamaIndex & AI Imports ---
from llama_index.core import Document, Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

# --- Pinecone & Pydantic ---
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field
from enum import Enum

# --- Configuration ---
load_dotenv()
PINECONE_INDEX_NAME = "job-search-index-small-llama"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INPUT_CSV_PATH = "job_dataset.csv" # IMPORTANT: Update this path
NUM_RECORDS_TO_PROCESS = 1000 # Process first 1000 for testing

# --- Pydantic Models for LLM Extraction ---
class JobTypeOptions(str, Enum):
    full_time = "full_time"
    part_time = "part_time"
    contract = "contract"
    internship = "internship"
    other = "other"

class CompanyTypeOptions(str, Enum):
    startup = "startup"
    big_tech = "big_tech"
    non_profit = "non_profit"
    government = "government"
    other = "other"

class ExtractedJobMetadata(BaseModel):
    """Structured metadata to be extracted from each job document."""
    skills_tags: List[str] = Field(description="A list of key skills, technologies, and experience levels mentioned.")
    job_type: JobTypeOptions = Field(description="The classified employment type based on the text.")
    company_type: CompanyTypeOptions = Field(description="The classified company type based on the text.")
    security_clearance: bool = Field(False, description="Whether a security clearance is likely required.")
    hourly_rate_min: Optional[float] = Field(None, description="The estimated minimum hourly salary, converted from annual if necessary.")
    hourly_rate_max: Optional[float] = Field(None, description="The estimated maximum hourly salary, converted from annual if necessary.")

# --- Helper to Clear Pinecone Index ---
def clear_pinecone_index(pc_client: Pinecone, index_name: str):
    """Deletes all vectors from the specified Pinecone index."""
    confirm = input(f"Are you sure you want to delete all data from the '{index_name}' index? This cannot be undone. (yes/no): ")
    if confirm.lower() != 'yes':
        print("Operation cancelled.")
        return False
    try:
        index = pc_client.Index(index_name)
        print(f"Clearing all data from index '{index_name}'...")
        index.delete(delete_all=True)
        print("Index cleared successfully.")
        return True
    except Exception as e:
        print(f"An error occurred while clearing the index: {e}")
        return False

# --- Main Ingestion Logic ---
def run_ingestion_pipeline():
    """
    Loads job data, enriches it using an LLM, and runs it through a LlamaIndex
    IngestionPipeline to store in Pinecone.
    """
    # 1. Initialize services using LlamaIndex Settings
    Settings.llm = OpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 2. Check if the Pinecone index exists, create it if not
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: '{PINECONE_INDEX_NAME}'")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # Dimension for text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    # Optional: Clear the index before starting
    if not clear_pinecone_index(pc, PINECONE_INDEX_NAME):
        return # Stop if user cancels deletion

    # 3. Load raw data from CSV
    try:
        df = pd.read_csv(INPUT_CSV_PATH).head(NUM_RECORDS_TO_PROCESS)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_CSV_PATH}'. Please update the path.")
        return
    
    # 4. Pre-process Data: Extract metadata first using an LLM program
    print("Stage 1: Extracting metadata from raw job data using an LLM...")
    
    # Define the LLM program for extraction
    extraction_prompt_template = """
    From the provided job posting text, extract the specified information.
    1. From the 'description', extract a list of key skills, technologies, and experience levels.
    2. Classify the 'job_type' into one of the following: full_time, part_time, contract, internship, other.
    3. Classify the 'company_type' based on the description into one of the following: startup, big_tech, non_profit, government, other.
    4. Determine if a 'security_clearance' is likely required (true/false).
    5. Assume a 2080-hour work year for any salary conversions.
    
    Job Posting:
    ---
    Title: {title}
    Description: {description}
    ---
    """
    program = LLMTextCompletionProgram.from_defaults(
        output_cls=ExtractedJobMetadata,
        prompt_template_str=extraction_prompt_template,
        llm=Settings.llm,
    )

    enriched_documents = []
    from tqdm import tqdm
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting Metadata"):
        try:
            # Run the LLM program to get structured metadata
            extracted_data: ExtractedJobMetadata = program(
                title=row.get("job_title", ""),
                description=row.get("job_description", "")
            )
            
            # Create the "Golden Document" text for embedding
            golden_text = f"Title: {row.get('job_title', '')}. "
            golden_text += f"Description: {row.get('job_description', '')}. "
            golden_text += f"Extracted Skills: {', '.join(extracted_data.skills_tags)}. "
            golden_text += f"Job Type: {extracted_data.job_type.value}. "
            
            # Generate a dummy timestamp within the last 90 days
            days_ago = random.randint(1, 90)
            dummy_date = datetime.now() - timedelta(days=days_ago)
            date_added_ts = int(dummy_date.timestamp())

            if extracted_data.hourly_rate_max is None:
                extracted_data.hourly_rate_max = extracted_data.hourly_rate_min

            base_metadata = {
                "title": row.get("job_title"),
                "organization": row.get("organization"),
                "location_literal": row.get("location"),
                "url": row.get("page_url"),
                **extracted_data.model_dump(), # Add all structured, extracted data
                "date_added_ts": date_added_ts,
            }

            metadata = {}
            for key, value in base_metadata.items():
                # Check for NaN specifically (which is a float) and also for None
                if isinstance(value, float) and math.isnan(value):
                    continue
                if value is None:
                    continue
                
                # Convert enums to their string values for Pinecone
                if isinstance(value, Enum):
                    metadata[key] = value.value
                else:
                    metadata[key] = value

            # Create the LlamaIndex Document
            doc = Document(
                text=golden_text,
                metadata=metadata
            )
            enriched_documents.append(doc)

        except Exception as e:
            print(f"Skipping a row due to extraction error: {e}")
            continue

    print(f"\nSuccessfully extracted metadata for {len(enriched_documents)} documents.")

    # 5. Define and run the Ingestion Pipeline
    print("\nStage 2: Running the Ingestion Pipeline (Chunking, Embedding, Storing)...")
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=20),
            Settings.embed_model, # The embedding model is a transformation
        ],
        vector_store=vector_store
    )

    # Run the pipeline with the enriched documents
    nodes = pipeline.run(documents=enriched_documents, show_progress=True)
    print(f"\nSuccessfully processed and ingested {len(nodes)} nodes into Pinecone.")

if __name__ == "__main__":
    run_ingestion_pipeline()
