# ==============================================================================
# FILE: visualize_dataset.py
#
# DESCRIPTION:
# This script loads the job dataset and generates several visualizations to
# provide insights into its composition. It creates plots for the top job
# sectors, locations, and job types.
#
# INSTRUCTIONS:
# 1. Make sure you have the required libraries installed:
#    pip install pandas matplotlib seaborn
# 2. Place your `monster_com-job_sample.csv` file in the same directory as this script.
# 3. Run the script: `python visualize_dataset.py`
# 4. A new directory named `visualizations` will be created with the plot images.
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    """Loads data and generates visualizations."""
    print("--- Starting Dataset Visualization Script ---")

    # --- 1. Load and Prepare Data ---
    try:
        df = pd.read_csv('monster_com-job_sample.csv')
        print(f"Successfully loaded {len(df)} rows from monster_com-job_sample.csv")
    except FileNotFoundError:
        print("Error: `monster_com-job_sample.csv` not found. Please place it in the same directory.")
        return

    # Create a directory to save the plots
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Set plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    # --- 2. Plot 1: Top 10 Job Sectors ---
    print("Generating plot for Top 10 Job Sectors...")
    plt.clf() # Clear the previous plot
    top_sectors = df['sector'].value_counts().nlargest(10)
    sns.barplot(x=top_sectors.values, y=top_sectors.index, palette="viridis")
    plt.title('Top 10 Job Sectors in the Dataset', fontsize=16)
    plt.xlabel('Number of Jobs', fontsize=12)
    plt.ylabel('Sector', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualizations/top_10_sectors.png')
    print("Saved 'top_10_sectors.png'")

    # --- 3. Plot 2: Top 10 Job Locations ---
    print("Generating plot for Top 10 Job Locations...")
    plt.clf() # Clear the previous plot
    # Clean up location data for better grouping
    df['location_cleaned'] = df['location'].str.split(',').str[0].str.strip()
    top_locations = df['location_cleaned'].value_counts().nlargest(10)
    sns.barplot(x=top_locations.values, y=top_locations.index, palette="plasma")
    plt.title('Top 10 Job Locations in the Dataset', fontsize=16)
    plt.xlabel('Number of Jobs', fontsize=12)
    plt.ylabel('Location', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualizations/top_10_locations.png')
    print("Saved 'top_10_locations.png'")

    # --- 4. Plot 3: Job Type Distribution ---
    print("Generating plot for Job Type Distribution...")
    plt.clf() # Clear the previous plot
    job_types = df['job_type'].dropna().astype(str).str.lower()
    
    # Standardize job types for better grouping
    def standardize_type(jt):
        if 'full' in jt: return 'Full-time'
        if 'part' in jt: return 'Part-time'
        if 'contract' in jt: return 'Contract'
        return 'Other'
        
    df['job_type_standardized'] = job_types.apply(standardize_type)
    
    sns.countplot(y=df['job_type_standardized'], order=df['job_type_standardized'].value_counts().index, palette="magma")
    plt.title('Distribution of Job Types', fontsize=16)
    plt.xlabel('Number of Jobs', fontsize=12)
    plt.ylabel('Job Type', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualizations/job_type_distribution.png')
    print("Saved 'job_type_distribution.png'")

    print("\n--- Visualization Script Finished ---")
    print("All plots have been saved to the 'visualizations' directory.")

if __name__ == "__main__":
    main()
