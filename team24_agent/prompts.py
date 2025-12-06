import textwrap

TASK_DESCRIPTION = textwrap.dedent("""
   Roleplay as the specific Yelp reviewer, whos profile is described below. Your goal is to step into their shoes and write a review for a new business they use/visit.

   [User Profile]
   {user}

   [Business Details]
   {business}

   {rag_context}
   
   [Logic Instructions]
   
   **Determine Rating:**
   Consider the user's profile, average rating, and past reviews on other similar businesses. Extract patterns on their strictness and preferences. 
   
   Analyze the business details and community reviews to gauge overall community sentiment on the business, and identify key strengths and weaknesses.

   Combine these insights to predict a star rating from 1.0 to 5.0 that the user would likely give this business. Some things of note include:
      - A strict user may tend to rate lower than the community average, and vice versa.
      - Specific dealbreaker attributes of a business, based on the user's preferences, can heavily impact the rating.
   
   **Draft Review:**
   Finally, write a review to accompany the predicted rating. Reference the user's previous reviews and mimic their vocabulary, tone, and writing style. Keep your reviews between 3-6 healthily lengthed sentences, on average, adjusting for the user's style.

   [Output Format (NO OTHER TEXT)]
   stars: [Rating 1.0-5.0]
   review: [Text]
""")