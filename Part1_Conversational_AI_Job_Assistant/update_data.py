# ==============================================================================
# FILE: update_pinecone_metadata.py
#
# DESCRIPTION:
# This script efficiently updates Pinecone records in batches. It first loads
# the entire CSV dataset into memory for fast lookups. Then, it fetches IDs
# from Pinecone in batches, finds the corresponding job descriptions from the
# in-memory data, adds a dummy date, and upserts the updated records back
# to Pinecone.
#
# INSTRUCTIONS:
# 1. Ensure your Pinecone index is already populated.
# 2. Set your PINECONE_API_KEY in a .env file.
# 3. Place your `job_dataset.csv` file in the same directory.
# 4. Run the script: `python update_pinecone_metadata.py`
# ==============================================================================

import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
import time
from datetime import datetime, timedelta
import random

# --- Configuration ---
load_dotenv()
PINECONE_INDEX_NAME = "job-search-index-large"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# --- Initialize Services ---
pc = Pinecone(api_key=PINECONE_API_KEY)

def generate_dummy_date() -> int:
    """Generates a random date within the last 30 days as a Unix timestamp."""
    days_ago = random.randint(0, 30)
    random_date = datetime.now() - timedelta(days=days_ago)
    return int(time.mktime(random_date.timetuple()))

def get_all_ids_from_index(index, namespace: str = "") -> list:
    """
    Fetches all vector IDs from the given Pinecone index.
    NOTE: For very large indexes, a more robust pagination strategy would be needed.
    """
    ids = set()
    # A common way to list IDs is to query with a dummy vector and a high top_k.
    # This might not return all IDs if the index is > 10,000.
    response = index.query(
        top_k=10000,
        vector=[0]*3072, # A zero vector is used to list IDs.
        namespace=namespace
    )
    for match in response.matches:
        ids.add(match.id)
    return list(ids)

def main():
    """Main function to run the update script."""
    print("--- Starting Pinecone Metadata Update Script ---")
    start_time = time.time()

    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        print(f"Successfully connected to index '{PINECONE_INDEX_NAME}'.")
    except Exception as e:
        print(f"Error connecting to Pinecone index: {e}")
        return

    # # --- 1. Load the dataset and create a composite key for fast lookups ---
    try:
        df = pd.read_csv('job_dataset.csv')
        # Use the fields necessary for the composite key + the data we need
        df = df[['page_url', 'job_title', 'job_description']].dropna()
        # Create a composite key for reliable matching
        df['lookup_key'] = df['page_url'].str.strip() +':::SPLITTER:::'+ df['job_title'].str.strip()
        df.set_index('lookup_key', inplace=True)
        print(f"Loaded {len(df)} rows from job_dataset.csv into memory.")
    except FileNotFoundError:
        print("Error: `job_dataset.csv` not found. Please place it in the root directory.")
        return
    except KeyError:
        print("Error: The CSV must contain 'page_url', 'job_title', and 'job_description' columns.")
        return

    # --- 2. Fetch all IDs from Pinecone ---
    print("Fetching all job IDs from Pinecone...")
    all_ids = get_all_ids_from_index(index)
    if not all_ids:
        print("No IDs found in the index. Exiting.")
        return
    print(f"Found {len(all_ids)} IDs to update.")

    # --- 3. Process and update records in batches ---
    batch_size = 100
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i+batch_size]
        print(f"Processing batch of {len(batch_ids)} (Rows {i+1}-{i+len(batch_ids)})...")

        # Fetch the full vectors for the current batch of IDs
        fetched_vectors = index.fetch(ids=batch_ids).vectors

        vectors_to_upsert = []
        for uniq_id, vector_data in fetched_vectors.items():
            try:
                # Reconstruct the lookup key from the Pinecone metadata
                page_url = vector_data.metadata.get('page_url', '').strip()
                job_title = vector_data.metadata.get('job_title', '').strip()
                lookup_key = page_url  + ':::SPLITTER:::' + job_title

                # Look up the description in the DataFrame using the composite key
                # Update the metadata with the new fields'
                job_description = df.loc[lookup_key, 'job_description']
                
                # Update the metadata with the new fields
                vector_data.metadata['job_description'] = str(job_description)
                vector_data.metadata['location_tags'] = [i.lower() for i in vector_data.metadata['location_tags']]
                # vector_data.metadata['date_added_ts'] = generate_dummy_date()
                
                # Reconstruct the full vector object for upserting
                vectors_to_upsert.append({
                    "id": uniq_id,
                    "values": vector_data.values,
                    "metadata": vector_data.metadata
                })
            except KeyError:
                # This happens if a job from Pinecone can't be found in the CSV
                print(f"  [Warning] Could not find a match for ID '{uniq_id}' in the CSV. Skipping.")
                continue
            except Exception as e:
                print(f"  [Error] An unexpected error occurred for ID '{uniq_id}': {e}. Skipping.")
                continue

        # Use the batched upsert command to overwrite the records
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
        
        print(f"Updated batch {i // batch_size + 1}.")
        time.sleep(0.5) # Be nice to the API

    end_time = time.time()
    print("\n--- Metadata Update Finished ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()