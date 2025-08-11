def search_jobs(preferences: JobPreferences) -> Dict[str, List[str]]:
    """Performs both a strict and relaxed search on Pinecone and returns job IDs."""
    if not preferences.role:
        return {"strict_ids": [], "relaxed_ids": []}

    query_text = f"A job posting for the role of : {preferences.role}"
    if preferences.keywords:
         query_text += f"with skills in : {', '.join(preferences.keywords)}"
         print(query_text)
    query_embedding = embeddings_model.embed_query(query_text)

    def build_positive_filter(is_strict: bool):
        filter_query = {}
        if preferences.location and preferences.location != "global":
            location_tags = get_location_tags(preferences.location)
            if "unknown" not in location_tags:
                filter_query["location_tags"] = {"$in": location_tags}
        if preferences.start_date_ts:
            filter_query["date_added_ts"] = {"$gte": preferences.start_date_ts, "$lte": preferences.end_date_ts}
        if preferences.salary_min:
            filter_query["hourly_rate_min"] = {"$gte": preferences.salary_min}
        if preferences.job_type:
            filter_query["job_type"] = {"$eq": preferences.job_type.value}
        return filter_query

    # Semantic exclusion for skills
    excluded_ids = set()
    if preferences.not_skills:
        positive_filter_for_exclusion = build_positive_filter(is_strict=False)
        exclusion_query_text = f"A job that is a {', '.join(preferences.not_skills)} role."
        exclusion_embedding = embeddings_model.embed_query(exclusion_query_text)
        # Find jobs semantically similar to the excluded skills
        excluded_results = index.query(
            vector=exclusion_embedding,
            filter=positive_filter_for_exclusion,
            top_k=100, # Get a pool of semantically similar jobs to exclude
            include_metadata=False
        )
        excluded_ids.update([res.id for res in excluded_results.matches])
    
    # --- Base Filter Construction ---
    def build_filter(is_strict: bool):
        filter_query = {}
        
        # Location
        if excluded_ids:
            filter_query["id"] = {"$nin": list(excluded_ids)}
            
        if preferences.location and preferences.location != "global":
            location_tags = get_location_tags(preferences.location)
            if "unknown" not in location_tags:
                filter_query["location_tags"] = {"$in": location_tags}
        
        # Date
        if preferences.start_date_ts:
            if is_strict:
                filter_query["date_added_ts"] = {
                    "$gte": preferences.start_date_ts, 
                    "$lte": preferences.end_date_ts
                }
            else:
                # Relaxed: Subtract 7 days to find slightly older jobs
                seven_days_in_seconds = 7 * 24 * 60 * 60
                relaxed_start_ts = preferences.start_date_ts - seven_days_in_seconds
                filter_query["date_added_ts"] = {
                    "$gte": relaxed_start_ts, 
                    "$lte": preferences.end_date_ts
                }
            
        # Salary
        if preferences.salary_min:
            if is_strict:
                filter_query["hourly_rate_min"] = {"$gte": preferences.salary_min}
            else:
                # Relaxed: Look for jobs paying up to 10% less
                relaxed_salary_min = preferences.salary_min * 0.9
                filter_query["hourly_rate_min"] = {"$gte": relaxed_salary_min}
            
        # Job Type
        if preferences.job_type:
            filter_query["job_type"] = {"$eq": preferences.job_type.value}

        # Company Type
        if preferences.company_type:
            filter_query["company_type"] = {"$in": [preferences.company_type.value]}

        # Security Clearance
        if preferences.security_clearance is not None:
             filter_query["security_clearance"] = {"$eq": preferences.security_clearance}

        # Keywords
        # if preferences.keywords:
            # # Add other keywords to the skills_tags filter
            # if "skills_tags" in filter_query and "$in" in filter_query["skills_tags"]:
            #      filter_query["skills_tags"]["$in"].extend([kw.lower().replace(' ', '_') for kw in preferences.keywords])
            # else:
            #      filter_query["skills_tags"] = {"$in": [kw.lower().replace(' ', '_') for kw in preferences.keywords]}
            
        # Negative Preferences
        if preferences.not_location:
            filter_query["location_literal"] = {"$nin": preferences.not_location}
            
        if preferences.not_skills:
            filter_query["skills_tags"] = {"$nin": [skill.lower().replace(' ', '_') for skill in preferences.not_skills]}
            
        return filter_query

    # --- Fetch a Larger Candidate Pool ---
    # We fetch more results than needed (e.g., 50) to have a good pool for re-ranking.
    print(excluded_ids)
    strict_filter = build_filter(is_strict=True)
    print(strict_filter)
    strict_ids = [res.id for res in index.query(vector=query_embedding, filter=strict_filter, top_k=50, include_metadata=False).matches]
    print(strict_ids)
    relaxed_filter = build_filter(is_strict=False)
    print(relaxed_filter)
    relaxed_ids = [res.id for res in index.query(vector=query_embedding, filter=relaxed_filter, top_k=50, include_metadata=False).matches]
    relaxed_ids = [rid for rid in relaxed_ids if rid not in strict_ids]
    print(relaxed_ids)
    # --- Hydrate the candidates to get metadata for sorting ---
    strict_jobs = hydrate_jobs(strict_ids)
    relaxed_jobs = hydrate_jobs(relaxed_ids)

    # --- Re-rank the results based on the sort_by preference ---
    if preferences.sort_by:
        if preferences.sort_by == SortOptions.highest_paying:
            # Sort by hourly_rate_max, descending. Handle missing salary by putting them last.
            strict_jobs.sort(key=lambda x: x.get('hourly_rate_max', -1), reverse=True)
            relaxed_jobs.sort(key=lambda x: x.get('hourly_rate_max', -1), reverse=True)
        elif preferences.sort_by == SortOptions.most_recent:
            # Sort by date_added_ts, descending.
            strict_jobs.sort(key=lambda x: x.get('date_added_ts', 0), reverse=True)
            relaxed_jobs.sort(key=lambda x: x.get('date_added_ts', 0), reverse=True)

    # Return the top 10 from the (potentially re-ranked) lists
    return {"strict_jobs": strict_jobs[:10], "relaxed_jobs": relaxed_jobs[:5]}


def hydrate_jobs(job_ids: List[str]) -> List[Dict]:
    """Fetch full job details from Pinecone metadata."""
    if not job_ids:
        return []
    fetched_data = index.fetch(ids=job_ids)
    return [vec.metadata for vec in fetched_data.vectors.values()]