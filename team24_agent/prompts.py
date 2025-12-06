import textwrap

TASK_DESCRIPTION = textwrap.dedent("""
    Roleplay as the specific user described below. Your goal is to simulate their authentic reaction to a new item.

    [User Profile]
    {user}

    [Business Details]
    {business}

    {rag_context}

    [Logic Instructions]
    1. **Determine Rating:** - **Calibrate:** Look at [User's Past Reviews]. Is this user a "Grumpy 2-star giver" or a "Happy 5-star giver"? 
      - **Evaluate:** Look at [Community Reviews]. Is the item actually good?
      - **Synthesize:** If the Community says "Good" but the User's History says "Hates this specific category," rate lower. If they align, rate normally.
      - **Decision:** Combine the User's Average Rating baseline with the specific merits of this item.
    
    2. **Draft Review:**
       - **Style:** Mimic the tone found in [User's Past Reviews]. 
       - **Content:** Mention specific attributes found in [Business Details] or [Community Reviews] that this user would notice.

    [Output Format]
    stars: [Rating 1.0-5.0]
    review: [Text]
""")