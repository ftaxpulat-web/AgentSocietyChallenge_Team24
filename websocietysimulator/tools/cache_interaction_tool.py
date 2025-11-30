import logging
import os
import json
import lmdb
from typing import Optional, Dict, List, Iterator
from tqdm import tqdm

logger = logging.getLogger("websocietysimulator")

class CacheInteractionTool:
    def __init__(self, data_dir: str):
        """
        Initialize the tool with the dataset directory.
        Args:
            data_dir: Path to the directory containing Yelp dataset files.
        """
        logger.info(f"Initializing InteractionTool with data directory: {data_dir}")
        self.data_dir = data_dir

        # Create LMDB environments
        self.env_dir = os.path.join(data_dir, "lmdb_cache")
        os.makedirs(self.env_dir, exist_ok=True)

        # Updated to handle large Yelp/Amazon datasets
        self.user_env = lmdb.open(os.path.join(self.env_dir, "users"), map_size=32 * 1024 * 1024 * 1024)
        self.item_env = lmdb.open(os.path.join(self.env_dir, "items"), map_size=16 * 1024 * 1024 * 1024)
        self.review_env = lmdb.open(os.path.join(self.env_dir, "reviews"), map_size=64 * 1024 * 1024 * 1024)

        # Initialize the database if empty
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the LMDB databases with data if they are empty."""
        # Imports added here for safety so you don't have to scroll up
        from collections import defaultdict 
        import json
        from tqdm import tqdm

        # 1. Initialize users
        # We check if the DB is empty first
        with self.user_env.begin(write=True) as txn:
            if not txn.stat()['entries']:
                print("Step 1/4: Processing Users...")
                with txn.cursor() as cursor:
                    # Added progress bar for Users
                    for user in tqdm(self._iter_file('user.json'), desc="Users"):
                        cursor.put(user['user_id'].encode(), json.dumps(user).encode())
            else:
                print("Users already cached. Skipping.")

        # 2. Initialize items
        with self.item_env.begin(write=True) as txn:
            if not txn.stat()['entries']:
                print("Step 2/4: Processing Items...")
                with txn.cursor() as cursor:
                    # Added progress bar for Items
                    for item in tqdm(self._iter_file('item.json'), desc="Items"):
                        cursor.put(item['item_id'].encode(), json.dumps(item).encode())
            else:
                print("Items already cached. Skipping.")

        # 3. Initialize reviews (OPTIMIZED: RAM Buffer)
        # We buffer indices in RAM to avoid the expensive read-modify-write loop
        item_review_index = defaultdict(list)
        user_review_index = defaultdict(list)

        with self.review_env.begin(write=True) as txn:
            if not txn.stat()['entries']:
                print("Step 3/4: Processing Reviews (Reading & Buffering)...")
                
                # Pass 1: Write Review Body and build memory index
                # Added progress bar for Reviews
                for review in tqdm(self._iter_file('review.json'), desc="Reading Reviews"):
                    # Store the review body
                    txn.put(review['review_id'].encode(), json.dumps(review).encode())
                    
                    # Store ID in memory buffer (Fast RAM operation)
                    item_review_index[review['item_id']].append(review['review_id'])
                    user_review_index[review['user_id']].append(review['review_id'])

                # Pass 2: Dump indices to LMDB
                print("Step 4/4: Saving Indices to Disk...")
                
                # Added progress bar for Item Index
                for item_id, review_ids in tqdm(item_review_index.items(), desc="Indexing Items"):
                    txn.put(f"item_{item_id}".encode(), json.dumps(review_ids).encode())
                
                # Added progress bar for User Index
                for user_id, review_ids in tqdm(user_review_index.items(), desc="Indexing Users"):
                    txn.put(f"user_{user_id}".encode(), json.dumps(review_ids).encode())
            else:
                print("Reviews already cached. Skipping.")

    def _iter_file(self, filename: str) -> Iterator[Dict]:
        """Iterate through file line by line."""
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield json.loads(line)

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Fetch user data based on user_id."""
        with self.user_env.begin() as txn:
            user_data = txn.get(user_id.encode())
            if user_data:
                return json.loads(user_data)
        return None

    def get_item(self, item_id: str) -> Optional[Dict]:
        """Fetch item data based on item_id."""
        if not item_id:
            return None

        with self.item_env.begin() as txn:
            item_data = txn.get(item_id.encode())
            if item_data:
                return json.loads(item_data)
        return None

    def get_reviews(
            self,
            item_id: Optional[str] = None,
            user_id: Optional[str] = None,
            review_id: Optional[str] = None
    ) -> List[Dict]:
        """Fetch reviews filtered by various parameters."""
        if review_id:
            with self.review_env.begin() as txn:
                review_data = txn.get(review_id.encode())
                if review_data:
                    return [json.loads(review_data)]
            return []

        with self.review_env.begin() as txn:
            if item_id:
                review_ids = json.loads(txn.get(f"item_{item_id}".encode()) or '[]')
            elif user_id:
                review_ids = json.loads(txn.get(f"user_{user_id}".encode()) or '[]')
            else:
                return []

            # Fetch complete review data for each review_id
            reviews = []
            for rid in review_ids:
                review_data = txn.get(rid.encode())
                if review_data:
                    reviews.append(json.loads(review_data))
            return reviews

    def __del__(self):
        """Cleanup LMDB environments on object destruction."""
        self.user_env.close()
        self.item_env.close()
        self.review_env.close()