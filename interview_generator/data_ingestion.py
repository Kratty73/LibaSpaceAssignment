import os
import pandas as pd
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
import time

# --- 1. Configuration and Initialization ---

# Load environment variables from .env file
load_dotenv()

# Securely get API keys
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not pinecone_api_key or not openai_api_key:
    print("Error: OpenAI or Pinecone API key not found in .env file.")
    exit()

# File paths and Pinecone settings
INPUT_QUESTIONS_JSON = "questions_output.json" # <-- CHANGED: Now reads from a JSON file
PINECONE_INDEX_NAME = "interview-question-bank"

print("Initializing clients...")
# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Initialize the embedding model (using a high-quality OpenAI model)
# You can switch to other models supported by LlamaIndex if needed.
embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=openai_api_key)
# Get embedding dimension
EMBEDDING_DIM = len(embed_model.get_text_embedding("test"))
print(f"Embedding dimension: {EMBEDDING_DIM}")


# --- 2. Data Loading and Preparation ---

def load_and_prepare_data():
    """Loads questions from the new JSON format and creates the rich text for embedding."""
    print("Loading and preparing data...")
    
    # Load enriched questions from the JSON file
    try:
        with open(INPUT_QUESTIONS_JSON, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Questions file not found at '{INPUT_QUESTIONS_JSON}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{INPUT_QUESTIONS_JSON}'. Please check the file format.")
        return None

    # --- Create the rich text string for each question ---
    documents_to_embed = []
    for question_obj in questions_data:
        # Extract data based on the new JSON structure
        question_text = question_obj.get('question_title', '')
        # The 'type' key from your JSON is used as the primary category and type
        question_type = question_obj.get('type', 'General') 
        difficulty = question_obj.get('difficulty', 'Medium')
        source_file = question_obj.get('source_file', 'Unknown')

        # Skip if the question title is missing
        if not question_text:
            continue
        
        # Combine all information into a single string for semantic embedding
        # The "||" separator helps the model distinguish the actual content from metadata
        rich_text = (
            f"Category: {question_type} | "
            f"Type: {question_type} | "
            f"Difficulty: {difficulty} || "
            f"{question_text}"
        )
        
        # Create a LlamaIndex Document object
        # We store the original question and its source in the metadata
        doc = Document(
            text=rich_text,
            metadata={
                "original_question": question_text,
                "source": source_file,
                "question_type": question_type, # Added for filtering
                "difficulty": difficulty        # Added for filtering
            }
        )
        documents_to_embed.append(doc)
        
    if not documents_to_embed:
        print("No valid documents were prepared for embedding. Please check your input file.")
        return None

    print(f"Successfully prepared {len(documents_to_embed)} documents for embedding.")
    print("\nExample of a rich text string being embedded:")
    print(f"'{documents_to_embed[0].text}'")
    print("\nExample of metadata being stored:")
    print(documents_to_embed[0].metadata)
    
    return documents_to_embed

# --- 3. Pinecone Index Setup ---

def setup_pinecone_index():
    """Checks if the Pinecone index exists and creates it if it doesn't."""
    print(f"\nSetting up Pinecone index: '{PINECONE_INDEX_NAME}'...")
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Index not found. Creating a new serverless index...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine", # Cosine similarity is great for semantic search
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" 
            )
        )
        # Wait for the index to be ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            print("Waiting for index to be ready...")
            time.sleep(5)
        print("Index created successfully.")
    else:
        print("Index already exists.")
        
    return pc.Index(PINECONE_INDEX_NAME)

# --- NEW: Function to clear the index ---
def clear_pinecone_index(pinecone_index):
    """Deletes all vectors from the specified Pinecone index."""
    print("\nClearing all existing vectors from the index...")
    # A common way to clear an index is to delete all vectors.
    # Pinecone's `delete` with `delete_all=True` is the standard way.
    try:
        pinecone_index.delete(delete_all=True)
        print("Index cleared successfully.")
        # Wait a moment for the deletion to propagate
        time.sleep(5) 
    except Exception as e:
        print(f"An error occurred while clearing the index: {e}")


# --- 4. LlamaIndex Ingestion Pipeline ---

def run_ingestion_pipeline(pinecone_index, documents):
    """Creates and runs a LlamaIndex pipeline to embed and upsert data."""
    print("\nStarting ingestion pipeline...")
    
    # Create a PineconeVectorStore instance
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    # Define the ingestion pipeline
    # It will take documents, run transformations (embedding), and store in the vector_store
    pipeline = IngestionPipeline(
        transformations=[
            embed_model,
        ],
        vector_store=vector_store,
    )
    
    # Run the pipeline
    pipeline.run(documents=documents, show_progress=True)
    
    print("Ingestion pipeline completed successfully.")


# --- Main Execution ---

if __name__ == "__main__":
    # Step 1: Prepare the data
    documents = load_and_prepare_data()
    
    if documents:
        # Step 2: Set up the Pinecone index
        pinecone_index = setup_pinecone_index()
        
        # Step 3: Clear the index before inserting new data
        clear_pinecone_index(pinecone_index)

        # Step 4: Run the ingestion pipeline
        run_ingestion_pipeline(pinecone_index, documents)
        
        # Step 5: Verify the number of vectors in the index
        stats = pinecone_index.describe_index_stats()
        print("\n--- Verification ---")
        print(f"Total vectors in index: {stats['total_vector_count']}")
        print("Setup complete. Your Pinecone index is ready for semantic search.")

