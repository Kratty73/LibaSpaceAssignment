import os
import json
import time
import logging
from tqdm import tqdm
import google.generativeai as genai

# --- Configuration ---
# Configure logging to see the script's progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Key & Model Setup ---
# Use the setup method from your reference code.
# This automatically looks for the GOOGLE_API_KEY environment variable.
if not os.getenv('GOOGLE_API_KEY'):
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to your API key.")
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize the model once, as shown in your reference.
model = genai.GenerativeModel('gemini-2.5-pro')

# --- Constants ---
SKILLS_FILE = 'skills.txt'
OUTPUT_FILE = 'ontology.json'
CHUNK_SIZE = 100  # Number of skills to process in a single API call
DELAY_BETWEEN_CALLS = 5  # Seconds to wait between API calls to respect rate limits

def get_parent_nodes_from_gemini(model, child_nodes_chunk, existing_parents):
    """
    Calls the Gemini API to get parent nodes for a chunk of child nodes.

    Args:
        model (genai.GenerativeModel): The initialized Gemini model instance.
        child_nodes_chunk (list): A list of child node objects to process.
        existing_parents (list): A list of existing parent node objects to reference.

    Returns:
        dict: A dictionary mapping child_id to a list of its parent's labels or IDs.
              Returns an empty dictionary on failure.
    """
    prompt = f"""
    You are a Knowledge Architect. Your task is to assign one or more immediate parent nodes to each skill in the given list. A skill can have multiple parents.

    Here is a list of parent nodes that have already been created. Reuse them if a skill fits logically.
    --- EXISTING PARENTS ---
    {json.dumps(existing_parents, indent=2)}
    ------------------------

    For each skill below, assign it to an existing parent's `id` or create a NEW parent by providing a descriptive label.
    --- SKILLS TO PROCESS ---
    {json.dumps(child_nodes_chunk, indent=2)}
    -------------------------

    Provide the output as a single, valid JSON object mapping each skill `id` to a LIST of its parent `id`(s) or new parent `label`(s).
    Example: {{ "skill_101": ["parent_backend_dev", "Data Engineering"] }}
    """

    try:
        # Use the model.generate_content() method as per your reference.
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except Exception as e:
        logging.error(f"An error occurred during API call: {e}")
        return {}


def main():
    """
    Main function to drive the ontology creation process.
    """
    logging.info("Starting ontology generation process...")

    # --- Step 1: Read initial skills ---
    if not os.path.exists(SKILLS_FILE):
        logging.error(f"{SKILLS_FILE} not found. Please create it with one skill per line.")
        # with open(SKILLS_FILE, 'w') as f:
        #     f.write("Python\nSQL\nReact\nProject Management\nLeadership\nAWS S3\nDocker\nKubernetes\nCSS\nHTML")
        # logging.info(f"Created a dummy {SKILLS_FILE} for demonstration.")
        
    with open(SKILLS_FILE, 'r') as f:
        skills = [line.strip() for line in f if line.strip()]

    # --- Step 2: Initialize data structures ---
    all_nodes = {}
    current_layer_ids = []
    node_counter = 0
    
    for skill in skills:
        node_id = f"skill_{node_counter}"
        all_nodes[node_id] = {
            "id": node_id,
            "label": skill,
            "type": "LEAF",
            "parent_ids": []
        }
        current_layer_ids.append(node_id)
        node_counter += 1

    logging.info(f"Initialized {len(skills)} leaf nodes.")

    # --- Step 3: Iteratively build the hierarchy ---
    layer_level = 1
    while len(current_layer_ids) > 1:
        logging.info(f"--- Processing Layer {layer_level} with {len(current_layer_ids)} nodes ---")
        
        next_layer_parent_nodes = {}
        existing_parents = []
        
        for i in tqdm(range(0, len(current_layer_ids), CHUNK_SIZE), desc=f"Layer {layer_level} Chunks"):
            chunk_ids = current_layer_ids[i:i + CHUNK_SIZE]
            chunk_nodes = [all_nodes[node_id] for node_id in chunk_ids]

            # Pass the initialized model to the function
            parent_mapping = get_parent_nodes_from_gemini(model, chunk_nodes, existing_parents)
            
            if not parent_mapping:
                logging.warning(f"Skipping chunk due to API error. Chunk start index: {i}")
                continue

            for child_id, parent_info_list in parent_mapping.items():
                if child_id not in all_nodes:
                    continue
                
                for parent_info in parent_info_list:
                    parent_id = None
                    if isinstance(parent_info, str) and parent_info.startswith("parent_"):
                        parent_id = parent_info
                    else:
                        parent_label = str(parent_info)
                        if parent_label in next_layer_parent_nodes:
                            parent_id = next_layer_parent_nodes[parent_label]['id']
                        else:
                            parent_id = f"parent_{node_counter}"
                            new_parent_node = {
                                "id": parent_id,
                                "label": parent_label,
                                "type": "INTERMEDIATE",
                                "parent_ids": []
                            }
                            all_nodes[parent_id] = new_parent_node
                            next_layer_parent_nodes[parent_label] = new_parent_node
                            existing_parents.append(new_parent_node)
                            node_counter += 1
                    
                    if parent_id and parent_id not in all_nodes[child_id]['parent_ids']:
                        all_nodes[child_id]['parent_ids'].append(parent_id)

            logging.info(f"Sleeping for {DELAY_BETWEEN_CALLS} seconds to respect rate limits...")
            time.sleep(DELAY_BETWEEN_CALLS)

        current_layer_ids = [p['id'] for p in next_layer_parent_nodes.values()]
        layer_level += 1

    # --- Step 4: Finalize and save the ontology ---
    if current_layer_ids:
        for root_id in current_layer_ids:
            if root_id in all_nodes:
                all_nodes[root_id]['type'] = 'ROOT'
        logging.info(f"Ontology converged to {len(current_layer_ids)} root node(s).")

    final_ontology = list(all_nodes.values())

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_ontology, f, indent=2)

    logging.info(f"Ontology generation complete! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
