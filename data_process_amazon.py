import json
import logging
import uuid
import pandas as pd
from tqdm import tqdm
import os
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REQUIRED_FILES_AMAZON = [
    'Industrial_and_Scientific.csv', 
    'Musical_Instruments.csv', 
    'Video_Games.csv',
    'Industrial_and_Scientific.jsonl', 
    'Musical_Instruments.jsonl', 
    'Video_Games.jsonl',
    'meta_Industrial_and_Scientific.jsonl', 
    'meta_Musical_Instruments.jsonl', 
    'meta_Video_Games.jsonl'
]



def load_data(file_path):
    """Load JSON data into a Pandas DataFrame with progress bar."""
    data = []
    with open(file_path, 'r') as file:
        for line in tqdm(file, desc=f"Loading {file_path}", unit=" lines"):
            data.append(json.loads(line))
    return pd.DataFrame(data)

def save_json(dataframe, output_file):
    """Save a Pandas DataFrame to a JSON file."""
    logging.info(f"Saving {output_file}...")
    dataframe.to_json(output_file, orient='records', lines=True)
    logging.info(f"{output_file} saved.")

def filter_data(top_cities, business_df, user_df, review_df):
    """Filter and save data within the top three cities with progress bars."""
    logging.info("Filtering data for top cities...")
    
    filtered_businesses = business_df[business_df['city'].isin(top_cities)]
    filtered_reviews = review_df[review_df['business_id'].isin(filtered_businesses['business_id'])]
    filtered_users = user_df[user_df['user_id'].isin(filtered_reviews['user_id'])]

    # Save filtered data in JSON format
    return filtered_businesses, filtered_reviews, filtered_users

def check_required_files(input_dir):
    """Check if all required files exist in the input directory."""
    all_required_files = REQUIRED_FILES_AMAZON 
    missing_files = []
    
    for file in all_required_files:
        if not os.path.exists(os.path.join(input_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print("Error: Missing required files:")
        for file in missing_files:
            print(f"- {file}")
        return False
    return True


def load_and_process_amazon_data(input_dir):
    """Load and process Amazon dataset."""
    logging.info("Loading and processing Amazon data...")
    rating_only_files = ['Industrial_and_Scientific.csv', 'Musical_Instruments.csv', 'Video_Games.csv']
    review_files = ['Industrial_and_Scientific.jsonl', 'Musical_Instruments.jsonl', 'Video_Games.jsonl']
    meta_files = ['meta_Industrial_and_Scientific.jsonl', 'meta_Musical_Instruments.jsonl', 'meta_Video_Games.jsonl']

    # Load rating-only data
    all_rating_only = pd.concat([pd.read_csv(os.path.join(input_dir, f)) for f in rating_only_files])
    users = all_rating_only['user_id'].unique().tolist()
    items = all_rating_only['parent_asin'].unique().tolist()

    # Load review data
    all_reviews = pd.DataFrame()
    for f in review_files:
        data = load_data(os.path.join(input_dir, f))
        # Filter data based on users and items from rating-only data
        data = data[data['user_id'].isin(users) & data['parent_asin'].isin(items)] 
        all_reviews = pd.concat([all_reviews, data])
    
    # Load meta data
    all_meta = pd.DataFrame()
    for f in meta_files:
        data = load_data(os.path.join(input_dir, f))
        data = data[data['parent_asin'].isin(items)]
        all_meta = pd.concat([all_meta, data])

    return all_reviews, all_meta



def merge_business_data(amazon_meta, output_file=None):
    """Merge business data from all sources while preserving source-specific columns."""
    logging.info("Merging business data for business...")
    
    
    # 将Amazon数据转换为json格式
    amazon_business = amazon_meta.rename(columns={
        'parent_asin': 'item_id'
    })
    amazon_business['source'] = 'amazon'
    amazon_business['type'] = 'product'
    amazon_json = json.loads(amazon_business.to_json(orient='records'))
    
  
    
    # 合并所有json数据
    merged_json = amazon_json 
    
    # 如果指定了输出文件，则保存
    if output_file:
        logging.info(f"Saving merged business data to {output_file}...")
        with open(output_file, 'w') as f:
            for item in merged_json:
                f.write(json.dumps(item) + '\n')

def merge_review_data(amazon_reviews, output_file=None):
    """Merge review data from all sources while preserving source-specific columns."""
    logging.info("Merging review data for reviews...")
    

    # 将Amazon评论转换为json格式
    amazon_reviews = amazon_reviews.rename(columns={
        'asin': 'sub_item_id',
        'parent_asin': 'item_id',
        'rating': 'stars',
    })
    amazon_reviews['review_id'] = [str(uuid.uuid4()) for _ in range(len(amazon_reviews))]
    amazon_reviews['source'] = 'amazon'
    amazon_reviews['type'] = 'product'
    amazon_json = json.loads(amazon_reviews.to_json(orient='records'))
    

    
    # 合并所有json数据
    merged_json = amazon_json
    # 如果指定了输出文件，则保存
    if output_file:
        logging.info(f"Saving merged review data to {output_file}...")
        with open(output_file, 'w') as f:
            for item in merged_json:
                f.write(json.dumps(item) + '\n')

def create_unified_users(amazon_reviews, output_file=None):
    """Create unified user data while preserving Yelp-specific user information."""
    logging.info("Merging users...")
    

    
    # 创建Amazon用户数据并转换为json格式
    amazon_users = pd.DataFrame({
        'user_id': amazon_reviews['user_id'].unique(),
        'source': 'amazon'
    })
    amazon_json = json.loads(amazon_users.to_json(orient='records'))
    

    # 合并所有json数据
    merged_json = amazon_json
    
    # 如果指定了输出文件，则保存
    if output_file:
        logging.info(f"Saving merged user data to {output_file}...")
        with open(output_file, 'w') as f:
            for item in merged_json:
                f.write(json.dumps(item) + '\n')

def main():
    """Main function with updated processing logic."""
    parser = argparse.ArgumentParser(description="Process multiple datasets for analysis.")
    parser.add_argument('--input_dir', required=True, help="Path to the input directory containing all dataset files.")
    parser.add_argument('--output_dir', required=True, help="Path to the output directory for saving processed data.")
    args = parser.parse_args()

    # Check required files
    if not check_required_files(args.input_dir):
        return


    
    # Process Amazon data
    amazon_reviews, amazon_meta = load_and_process_amazon_data(args.input_dir)
    
    # Merge all data
    os.makedirs(args.output_dir, exist_ok=True)
    merge_business_data(amazon_meta, os.path.join(args.output_dir, 'item.json'))
    merge_review_data(amazon_reviews, os.path.join(args.output_dir, 'review.json'))
    create_unified_users(amazon_reviews, os.path.join(args.output_dir, 'user.json'))

    logging.info("Data processing completed successfully.")

if __name__ == '__main__':
    main()