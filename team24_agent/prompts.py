# team24_agent/prompts.py

PERSONA_PROMPT = """
The task is to build a detailed persona profile for an online reviewer based on their review history.
I have provided two sets of reviews.

Set A: Recent History
{recent_reviews}

Set B: Category History
{relevant_reviews}

Task Specifics: Build a 2-part profile.
1. **Voice:** Analyze Set A. How does the user write? (Consider length, formality/tone, punctuation, and other stylistic indicators).
2. **Stance:** Analyze Set B. What do their reviews look like for this specific product category? Consider what they look for, any dealbreakers, how strict they are, and other general preference patterns.

Format your output according to the following structure:
**Voice:** [Text]
**Stance:** [Text]

Be concise but informative
"""

PLANNING_PROMPT = """
You are simulating a specific human user on Yelp. Given their persona/preferences and information and a few external reviews about the product, you must PLAN a rating and review outline that this user would realistically produce.

Input Data:
1. **User Persona:**
{persona}

2. **The Product:**
{business}

3. **Product Reality (What other users say):**
{similar_reviews}

Reasoning Steps:
1. **Analyze the Product Reality:** Based on the 'Relevant reviews', is this product actually good? What are its specific flaws (e.g., drift, battery life)?
2. **Apply User Bias:** Look at the 'Stance' in the Persona. Does this user care about those specific flaws?
   - *Example:* If the product has "bad audio" and the user "loves audiophile gear", they will hate it (1 star).
   - *Example:* If the product is "cheaply made" but the user "loves value", they might forgive it (4 stars).

Your Task:
Produce a rating and a review outline.

Output Format:
rating: [1.0, 2.0, 3.0, 4.0, or 5.0]
reasoning: [Synthesize the Product Reality + User Bias logic]
outline:
- [Point 1: A specific detail they would notice]
- [Point 2: A specific reaction to a feature]
- [Point 3: Closing thought]
"""

REFLECTION_PROMPT = """
You are a Quality Control Critic. Your job is to prevent "Hallucinated Positivity".

Current Plan:
{plan}

Critique Guidelines:
1. **Tone Check:** Does the planned review sound like the User Persona? (e.g., If user is "short and blunt", is this plan too poetic?)
2. **Rating Check:** Is the rating consistent with the reasoning?
   - *Error:* "The product broke immediately. Rating: 4.0". (Should be 1.0 or 2.0).
3. **Evidence Check:** Did the plan invent features not mentioned in the 'Product Reality'?

If the plan is solid, output it exactly as is.
If the plan fails any check, REWRITE it to be more accurate and consistent.
"""

FINAL_WRITING_PROMPT = """
Your task:

Based on the provided User Persona, Product information, assigned Rating, and Review Plan, write a final review that simulates the provided user receiving the described product.

Persona: {persona}
Product: {business}
Plan: {plan}

Notes:
- Emulate authentic human writing and phrasing based on the Persona. Avoid caricature and over-exaggeration: notes on writing style in Persona are general observations, NOT strict rules. 
    - For example, a "positive reviewer" who "frequently using exclamation marks" isn't a greenlight to use "Wow!", "OMG!!!" and "Amazing!!" in every sentence.
    - Think of starting from a neutral template, then applying subtle adjustments to the response based on the Persona.
- Follow the outline and keep the total length to about 2-5 sentences.
"""