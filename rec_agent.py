import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
import re
import logging
import time
logging.basicConfig(level=logging.INFO)

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except:
        print(encoding.encode(string))
    return a

class RecPlanning(PlanningBase):
    """Inherits from PlanningBase"""
    
    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)
    
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Override the parent class's create_prompt method"""
        if feedback == '':
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

Task: {task_description}
'''
            prompt = prompt.format(task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        return prompt

class RecMemory:
    """
    Extremely compact, per-task memory for user preferences.

    - Input: raw review text (possibly long, noisy).
    - Output: exactly 5 sentences summarizing the user's preferences.
    - Stores a single profile string for the current task.
    """

    def __init__(self, llm, max_sentences: int = 5):
        self.llm = llm
        self.max_sentences = max_sentences
        self._user_profile: str = ""

    def clear(self):
        """Reset memory at the beginning of each new task."""
        self._user_profile = ""

    def __call__(self, review_text: str = "") -> str:
        """
        If review_text is provided and non-empty:
          - Summarize reviews into a 5-sentence user profile and store it.
        If review_text is empty:
          - Return the stored profile (may be "" if not set yet).
        """
        if review_text:
            cleaned = review_text.strip()
            print("RecMemory.__call__ got review_text:", repr(cleaned)[:200])
            if cleaned and cleaned not in ("[]", "{}", "None"):
                self._user_profile = self._build_profile(cleaned)
            else:
                # Optional: log once for debugging
                print("RecMemory: review_text is empty / uninformative, skipping LLM.")
        return self._user_profile

    def _build_profile(self, review_text: str) -> str:
        """
        Use the LLM to summarize review_text into a compact profile,
        then enforce exactly max_sentences sentences.
        """
        prompt = f"""
You are building a concise user preference profile from their historical reviews.

You will be given multiple reviews (possibly noisy). Your job:
- Extract stable, high-level preferences (e.g., favorite categories, brands, styles, quality/price tradeoffs).
- Ignore one-off outliers or random noise.
- Focus on what this user tends to LIKE and DISLIKE across items.

Write EXACTLY {self.max_sentences} sentences.
- Each sentence should be complete and end with a period.
- Do NOT number the sentences.
- Do NOT include bullet points or headings.

User review history:
{review_text}
"""

        messages = [{"role": "user", "content": prompt}]
        try:
            raw = self.llm(
                messages=messages,
                temperature=0.1,
                max_tokens=4096,
            )
        except Exception as e:
            print("RecMemory: error while summarizing reviews:", repr(e))
            return ""
        sentences = self._to_exact_n_sentences(raw, self.max_sentences)
        print("RecMemory: prompt:", prompt)
        print("RecMemory: built user profile:", sentences)
        return sentences

    def _to_exact_n_sentences(self, text: str, n: int) -> str:
        """
        Best-effort post-processing to ensure exactly n sentences.
        Splits on '.', '!' or '?' and recombines.
        """
        if not text:
            return ""

        # Rough split into sentences
        import re as _re
        parts = _re.split(r'([.!?])', text)
        sentences = []
        current = ""

        for part in parts:
            if not part:
                continue
            current += part
            if part in ".!?":
                s = current.strip()
                if s:
                    sentences.append(s)
                current = ""
        tail = current.strip()
        if tail:
            sentences.append(tail if tail.endswith(('.', '!', '?')) else tail + ".")

        if not sentences:
            return ""
        if len(sentences) >= n:
            return " ".join(sentences[:n])
        last = sentences[-1]
        while len(sentences) < n:
            sentences.append(last)
        return " ".join(sentences)


class RecReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""

    def __init__(self, profile_type_prompt, llm, memory: RecMemory):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=memory, llm=llm)
        self.memory = memory

    def __call__(self, task_description: str):
        """
        task_description already includes:
          - candidate_list
          - user history
          - item info
        We now ALSO include the 5-sentence user profile from memory,
        and ask the model to output per-item scores in JSON.
        """
        user_profile = ""
        if self.memory is not None:  # retrieve the stored profile
            user_profile = self.memory() or ""
        if user_profile:
            profile_block = user_profile
        else:
            profile_block = (
                "No reliable user review history is available. "
                "Treat the user as a cold-start user and rely on item features "
                "and general appeal only."
            )

        prompt = f"""
You are a recommendation ranking model.

User preference summary (5 sentences):
{profile_block}

Task:
- For this user, assign a relevance score from 0.0 to 10.0 to EACH candidate item ID.

Scoring rules (IMPORTANT):
- 10.0 = extremely relevant and strongly preferred.
- 0.0 = extremely irrelevant for this user.
- Most items should be somewhere in between.
- Do NOT give the same score to all items.
- Items that are more relevant MUST have higher scores than less relevant items.
- Try to use the full range [0.0, 10.0] across the candidate list.

Constraints:
- Use ONLY the item IDs in the provided candidate list.
- Include ALL of those IDs exactly once.
- Do NOT invent new IDs.
- Do NOT omit any IDs.
- Do NOT include any explanation or commentary.

Output format (VERY IMPORTANT):
- Output ONLY valid JSON, no backticks or extra text.
- The JSON MUST be an array of objects with "item_id" and "score" fields.
- Example:
[
  {{"item_id": "ITEM_ID_1", "score": 9.5}},
  {{"item_id": "ITEM_ID_2", "score": 6.0}},
  {{"item_id": "ITEM_ID_3", "score": 2.0}}
]

Context for scoring:
{task_description}
""".strip()


        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=8192,
        )
        return reasoning_result




class MyRecommendationAgent(RecommendationAgent):
    """
    Participant's implementation of SimulationAgent, updated to
    incorporate winning-strategy ideas from baseline666, RecHackers,
    and DummyAgent:
      - platform-aware item feature engineering
      - platform-aware review filtering
      - prompt that emphasizes informative user + item text
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.memory = RecMemory(llm=llm, max_sentences=5)
        self.reasoning = RecReasoning(profile_type_prompt='', llm=self.llm, memory=self.memory)
        self.platform: str = "unknown"  # "yelp" | "amazon" | "goodreads" | "unknown"

    # Helpers: platform & feature engineering 

    def _detect_platform_from_item(self, item: dict) -> str:
        """
        Heuristic detection of the platform based on item fields.
        Matches the high-level descriptions in the AgentSociety paper:
        - Amazon: title, description, average_rating, rating_number, etc.
        - Goodreads: title_without_series, authors, similar_books, ratings_count, etc.
        - Yelp: name, stars, review_count, categories, etc.
        """
        if not isinstance(item, dict):
            return "unknown"

        # Goodreads
        if (
            "title_without_series" in item
            or "similar_books" in item
            or "authors" in item
        ):
            return "goodreads"

        # Amazon
        if (
            "description" in item
            or "average_rating" in item
            or "rating_number" in item
        ):
            return "amazon"

        # Yelp
        if (
            "name" in item
            or "stars" in item
        ):
            return "yelp"

        return "unknown"

    def _extract_item_features(self, item: dict, platform: str) -> dict:
        """
        Platform-specific item-side feature engineering (baseline666-style).
        Only keeps fields that are most informative for ranking.
        """

        if not isinstance(item, dict):
            return {}

        base_keys = ["item_id"]

        if platform == "amazon":
            # Product features
            extra_keys = [
                "title",              # product name
                "average_rating",
                "rating_number",      # review count
                "description",
                "brand",
                "categories",
                "price",
            ]
        elif platform == "yelp":
            # Business features
            extra_keys = [
                "name",
                "stars",
                "review_count",
                "categories",
                "city",
                "state",
            ]
        elif platform == "goodreads":
            # Book features
            extra_keys = [
                "title_without_series",
                "authors",
                "publication_year",
                "ratings_count",
                "average_rating",
                "similar_books",
                "genres",
            ]
        else:
            # Default
            extra_keys = [
                "name",
                "title",
                "stars",
                "review_count",
                "average_rating",
                "rating_number",
                "description",
            ]

        keys_to_keep = base_keys + extra_keys
        filtered = {k: item[k] for k in keys_to_keep if k in item}
        if "description" in filtered:
            filtered["description"] = str(filtered["description"])[:100]
        return filtered

    def _normalize_review_list(self, reviews_raw):
        """
        Turn whatever InteractionTool returns into a list[dict].
        The tool may return a dict-of-dicts or an already-formed list.
        """
        if reviews_raw is None:
            return []

        if isinstance(reviews_raw, list):
            return [r for r in reviews_raw if isinstance(r, dict)]

        if isinstance(reviews_raw, dict):
            return [v for v in reviews_raw.values() if isinstance(v, dict)]

        return []

    def _score_review(self, r: dict, platform: str) -> float:
        """
        Platform-specific scoring of reviews to select the most informative ones.
        Mirrors the DummyAgent-style focus on 'useful/funny/cool' (Yelp),
        'verified purchase + date' (Amazon), and 'votes/comments/reading_status'
        (Goodreads), but coded defensively since field names may vary.
        """
        if not isinstance(r, dict):
            return 0.0

        score = 0.0

        # Positive signals
        for key in ["useful", "funny", "cool", "votes", "vote",
                    "helpful", "helpful_votes", "n_votes"]:
            val = r.get(key, 0) or 0
            if isinstance(val, (int, float)):
                score += float(val)

        # Rating indicates strength
        for key in ["stars", "rating", "review_rating"]:
            val = r.get(key, None)
            if isinstance(val, (int, float)):
                score += 0.1 * float(val)

        if platform == "amazon":
            # Verified bonus
            if r.get("verified_purchase") or r.get("verified", False):
                score += 1.0

        if platform == "goodreads":
            # Read reviews more important
            if r.get("read_status") in ["read", "currently-reading"]:
                score += 1.0
            # Comments / interactions 
            for key in ["n_comments", "comments_count"]:
                val = r.get(key, 0) or 0
                if isinstance(val, (int, float)):
                    score += 0.5 * float(val)

        # Timestamp recency bonus score
        for key in ["date", "review_date", "timestamp", "time"]:
            val = r.get(key, None)
            if isinstance(val, (int, float)):
                score += 0.00000001 * float(val)

        return score

    def _select_informative_reviews(
        self,
        reviews_raw,
        platform: str,
        max_reviews: int = 20
    ) -> list[dict]:
        """
        Take all user reviews and keep the most informative ones,
        sorted by a platform-aware score, DummyAgent-style.
        """
        reviews = self._normalize_review_list(reviews_raw)
        if not reviews:
            return []

        scored = [
            (self._score_review(r, platform), r)
            for r in reviews
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [r for _, r in scored[:max_reviews]]
        return selected

    def _format_reviews_for_prompt(self, reviews: list[dict], platform: str) -> str:
        """
        Convert structured reviews into a compact text block for the LLM.
        We only keep a few key fields plus text to avoid blowing up tokens.
        """
        lines = []
        for r in reviews[:20]:
            if not isinstance(r, dict):
                continue

            # Common fields
            text = str(
                r.get("text")
                or r.get("review_text")
                or r.get("body")
                or ""
            ).strip()

            rating = r.get("stars") or r.get("rating") or r.get("review_rating")
            date = r.get("date") or r.get("review_date") or r.get("timestamp")

            meta_parts = []
            if rating is not None:
                meta_parts.append(f"rating={rating}")
            if date is not None:
                meta_parts.append(f"date={date}")

            if platform == "amazon" and r.get("verified_purchase"):
                meta_parts.append("verified_purchase=True")
            if platform == "yelp":
                for k in ["useful", "funny", "cool"]:
                    if k in r:
                        meta_parts.append(f"{k}={r[k]}")
            if platform == "goodreads":
                if "read_status" in r:
                    meta_parts.append(f"status={r['read_status']}")
                for k in ["n_votes", "n_comments"]:
                    if k in r:
                        meta_parts.append(f"{k}={r[k]}")

            meta_str = "; ".join(meta_parts) if meta_parts else ""
            if meta_str:
                lines.append(f"[{meta_str}] {text}")
            else:
                lines.append(text)

        return "\n".join(lines)

    # Main workflow 

    def workflow(self):
        """
        Largely follows the standard winning-agent workflow:
          1) query user info
          2) query candidate item info (platform-specific features)
          3) query & filter user reviews (platform-specific selection)
          4) prompt LLM with user history + item info + candidate list
        Returns:
            list: Sorted list of item IDs
        """
        if hasattr(self, "memory") and self.memory is not None:
            self.memory.clear()

        plan = [
            {'description': 'First I need to find user information'},
            {'description': 'Next, I need to find item information'},
            {'description': 'Next, I need to find review information'}
        ]

        user_text = ''
        item_list = []
        history_review_text = ''

        platform = "unknown"

        for sub_task in plan:
            desc = sub_task.get('description', '').lower()

            if 'user' in desc:
                # Raw user info 
                user_info = self.interaction_tool.get_user(
                    user_id=self.task['user_id']
                )
                user_text = str(user_info)
                if len(user_text) > 2000:
                    user_text = user_text[:2000]

            elif 'item' in desc:
                # Item info with platform-specific feature engineering
                for item_id in self.task['candidate_list']:
                    item = self.interaction_tool.get_item(item_id=item_id)

                    if item is None:
                        print(f"[WARN] get_item returned None for item_id={item_id}, skipping.")
                        continue

                    # Detect platform from first non-null item
                    if platform == "unknown":
                        platform = self._detect_platform_from_item(item)

                    filtered_item = self._extract_item_features(
                        item, platform=platform
                    )

                    filtered_item.setdefault("item_id", item_id)
                    item_list.append(filtered_item)

            elif 'review' in desc:
                # Fetch all user reviews, then filter 
                raw_reviews = self.interaction_tool.get_reviews(
                    user_id=self.task['user_id']
                )
                print("DEBUG raw_reviews for user", self.task['user_id'], "->", type(raw_reviews), repr(raw_reviews)[:500])
                selected_reviews = self._select_informative_reviews(
                    raw_reviews,
                    platform=platform,
                    max_reviews=20
                )
                history_review_text = self._format_reviews_for_prompt(
                    selected_reviews,
                    platform=platform
                )

                if not history_review_text:
                    fallback = str(raw_reviews)
                    input_tokens = num_tokens_from_string(fallback)
                    if input_tokens > 12000:
                        encoding = tiktoken.get_encoding("cl100k_base")
                        fallback = encoding.decode(
                            encoding.encode(fallback)[:12000]
                        )
                    history_review_text = fallback

                if history_review_text and self.memory is not None:
                    self.memory(history_review_text)


            else:
                pass

        # Final prompt: emphasize platform, user history, item features

        candidate_list = self.task['candidate_list']
        platform_str = {
            "yelp": "Yelp (local business reviews)",
            "amazon": "Amazon (e-commerce products)",
            "goodreads": "Goodreads (books & reading)",
            "unknown": "an online platform",
        }.get(platform, "an online platform")

        platform_str = {
            "yelp": "Yelp",
            "amazon": "Amazon",
            "goodreads": "Goodreads",
            "unknown": "Unknown",
        }.get(platform, "Unknown")

        task_description = f"""
PLATFORM:
{platform_str}

CANDIDATE_IDS:
{candidate_list}

USER_HISTORY:
{history_review_text}

USER_INFO:
{user_text}

CANDIDATE_ITEMS_INFO:
{item_list}
""".strip()
        print("Final task_description for reasoning:", task_description)
        result = self.reasoning(task_description)

        candidate_list = list(self.task.get("candidate_list", []))

        # --- New: parse JSON scores and sort in Python ---
        try:
            import json

            # Try to find a JSON array in the output (in case the model adds stray text)
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if not match:
                print("No JSON array found in LLM output. Falling back to candidate_list.")
                print("Raw LLM output:", result)
                return candidate_list

            json_str = match.group()
            parsed = json.loads(json_str)

            if not isinstance(parsed, list):
                print("Parsed JSON is not a list. Falling back to candidate_list.")
                print("Parsed:", parsed)
                return candidate_list

            # Normalize and filter: keep only valid items with an id in candidate_list
            scored_items = []
            for obj in parsed:
                if not isinstance(obj, dict):
                    continue
                item_id = str(obj.get("item_id", "")).strip()
                if item_id not in candidate_list:
                    continue
                score_val = obj.get("score", 0.0)
                try:
                    score = float(score_val)
                except (TypeError, ValueError):
                    score = 0.0
                scored_items.append((item_id, score))

            if not scored_items:
                print("No valid (item_id, score) pairs after filtering. Falling back to candidate_list.")
                print("Raw LLM output:", result)
                print("Parsed JSON:", parsed)
                return candidate_list

            # Sort by score descending
            scored_items.sort(key=lambda x: x[1], reverse=True)

            # Extract ranking
            ranked_ids = [item_id for (item_id, _) in scored_items]

            # For safety: ensure all candidate IDs are present (append missing in original order)
            missing = [cid for cid in candidate_list if cid not in ranked_ids]
            ranked_ids.extend(missing)

            print("Raw LLM result:", result)
            print("JSON_str:", json_str)
            print("Parsed JSON:", parsed)
            print("Scored_items:", scored_items)
            print("Final ranked_ids:", ranked_ids)

            return ranked_ids

        except Exception as e:
            print("Format error when parsing JSON scores:", repr(e))
            print("Raw LLM output:", result)
            return candidate_list




from typing import List, Dict, Any, Optional
import ast

class DummyEmbeddingModel:
    """
    Minimal embedding model stub so that parts of the framework
    that expect embed_documents/embed_query don't crash.
    It just returns constant vectors.
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if texts is None:
            return []
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.0]

from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types  # for config, if you want
import os
from dotenv import load_dotenv

class GeminiLLM(LLMBase):
    """
    Real Gemini-backed LLM for the recommendation agent.
    Wraps the official google-genai client.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",  # or "gemini-3.0-pro" etc.
        api_key: Optional[str] = None,
    ):
        # LLMBase expects a model name string
        super().__init__(model=model)

        # Create Gemini client. If api_key is None, it will read GEMINI_API_KEY
        # from the environment.
        load_dotenv()
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set and no api_key passed to GeminiLLM")
        self._client = genai.Client(api_key=api_key)

        # You can reuse your dummy embedding model until you actually
        # want real embeddings.
        self._embedding_model = DummyEmbeddingModel()

    def _extract_text_from_response(self, response) -> str:
        """
        Safely extract concatenated text from a GenerateContentResponse.
        Returns "" if there is no text at all.
        """
        if response is None or not getattr(response, "candidates", None):
            return ""

        texts = []
        for cand in response.candidates:
            content = getattr(cand, "content", None)
            if content is None:
                continue

            parts = getattr(content, "parts", None)
            if not parts:
                # e.g., when finish_reason == MAX_TOKENS before any text
                continue

            for part in parts:
                t = getattr(part, "text", None)
                if t:
                    texts.append(t)

        return "\n".join(texts).strip()

    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        stop_strs: Optional[List[str]] = None,
        n: int = 1,
    ) -> str:
        if not messages:
            return ""

        # Flatten chat messages into a single text prompt
        prompt_lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt_lines.append(f"{role.upper()}: {content}")
        prompt_text = "\n".join(prompt_lines)

        try:
            response = self._client.models.generate_content(
                model=model or self.model,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
        except Exception as e:
            print("GeminiLLM error:", repr(e))
            return ""

        # DEBUG: see raw response and finish_reason
        print("Gemini raw response:", repr(response))
        for cand in response.candidates:
           print("Finish reason:", getattr(cand, "finish_reason", None))

        text = self._extract_text_from_response(response)

        if not text:
            # Explicitly log instead of silently returning "['']"
            print("GeminiLLM: no text extracted from response, returning empty string.")
            return ""

        return text

if __name__ == "__main__":
    task_set = "amazon"  # "goodreads" or "yelp"
    # Initialize Simulator
    simulator = Simulator(data_dir="./amazon_data_processed", device="auto", cache=False)

    # Load scenarios
    simulator.set_task_and_groundtruth(
        task_dir=f"./example/track2/{task_set}/tasks",
        groundtruth_dir=f"./example/track2/{task_set}/groundtruth",
    )

    # Set your custom agent
    simulator.set_agent(MyRecommendationAgent)

    # Set LLM client
    simulator.set_llm(GeminiLLM(model="gemini-2.5-flash"))

    # --- STEP 1: test run_simulation only ---
    print("About to run simulation...")
    agent_outputs = simulator.run_simulation(
        number_of_tasks=25, enable_threading=False, max_workers=1
    )
    print("Simulation finished. ") # Sample output for first task:", agent_outputs[:1])

    # --- STEP 2: test evaluate separately ---
    print("About to evaluate...")
    evaluation_results = simulator.evaluate()
    import math
    import glob

    # Collect ground-truth labels in order
    gt_items = []  # ground-truth item_id per scenario

    # Assuming groundtruth is a bunch of jsonlines or jsons in groundtruth_dir
    gt_files = sorted(glob.glob(f"./example/track2/{task_set}/groundtruth/*.json*"))

    for gt_path in gt_files[:len(agent_outputs)]:
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_obj = json.load(f)

        gt_item_id = gt_obj.get("ground truth")

        if gt_item_id is None:
            # If schema different, print once to help debug
            print("WARNING: groundtruth schema unknown for", gt_path, "object:", gt_obj)
            gt_item_id = ""  # fallback; will be treated as "missing" below

        gt_items.append(gt_item_id)

    for i, scenario in enumerate(agent_outputs):
        task = scenario["task"]
        pred_list = scenario["output"]
        cand_list = task["candidate_list"]
        gt_id = gt_items[i]

        print(f"Scenario {i}")
        print("  GT item:", gt_id)
        print("  In candidate_list?", gt_id in cand_list)
        print("  In pred_list?", gt_id in pred_list)
        print()


    # Compute RMSE of rank (1 = best possible)
    squared_errors = []
    for idx, (pred, gt_id) in enumerate(zip(agent_outputs, gt_items)):
        # agent_outputs element can be either a dict with 'output' or already a list
        if isinstance(pred, dict) and "output" in pred:
            pred_list = pred["output"]
        else:
            pred_list = pred

        if not isinstance(pred_list, list):
            print(f"WARNING: prediction for scenario {idx} has unexpected type:", type(pred_list), pred_list)
            continue
        if gt_id not in pred_list:
            print(f"[DEBUG] GT id {gt_id} not in prediction list for scenario {idx}")

        if gt_id in pred_list:
            # rank index (1-based)
            rank = pred_list.index(gt_id) + 1
        else:
            # Penalize missing item with "worst" rank (len+1)
            rank = len(pred_list) + 1

        error = (rank - 1) ** 2  # target rank = 1
        squared_errors.append(error)

    if squared_errors:
        rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
    else:
        rmse = None

    # Attach RMSE into evaluation_results so it appears in the JSON
    if "metrics" not in evaluation_results:
        evaluation_results["metrics"] = {}
    evaluation_results["metrics"]["rmse_rank_1based"] = rmse
    print("Custom RMSE (rank vs 1):", rmse)

    # --- STEP 4: save results with RMSE ---
    import os
    os.makedirs("./results", exist_ok=True)
    with open(f'./results/evaluation_results_track2_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print("Done. Saved evaluation_results with RMSE.")