import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Base destination folder for cropped outputs
OUTPUT_DIR = "data/processed_data/"

def process_ctsd(base_path, test_size=0.2, random_state=42):
    """Parses annotations, performs stratified train/test split, and crops CTSD."""
    csv_path = os.path.join(base_path, "annotations.csv")
    images_dir = os.path.join(base_path, "images/")
    
    # Load dataset
    a = pd.read_csv(csv_path)
    
    # Filter out invalid bounding boxes
    valid_boxes = (a['x2'] > a['x1']) & (a['y2'] > a['y1'])
    a = a[valid_boxes].dropna().reset_index(drop=True)
    
    # Stratified Train/Test Split (preserves class distribution across splits)
    train_df, test_df = train_test_split(
        a, test_size=test_size, stratify=a['category'], random_state=random_state
    )
    
    def crop_and_save(df, split_type):
        print(f"Processing CTSD [{split_type}]...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            img_path = os.path.join(images_dir, row['file_name'])
            if not os.path.exists(img_path):
                continue
            
            # Destination: /OUTPUT_DIR/CTSD///
            save_dir = os.path.join(OUTPUT_DIR, "CTSD", split_type, str(int(row['category'])))
            os.makedirs(save_dir, exist_ok=True)
            
            with Image.open(img_path) as img:
                # PIL crop box format: (left, upper, right, lower)
                crop_box = (row['x1'], row['y1'], row['x2'], row['y2'])
                cropped_img = img.crop(crop_box)
                
                save_path = os.path.join(save_dir, f"{idx}_{row['file_name']}")
                cropped_img.save(save_path)

    crop_and_save(train_df, "train")
    crop_and_save(test_df, "test")  

def process_gtsrb(base_path):
    """Processes GTSRB train and test CSV files and crops ROIs."""
    
    def crop_and_save(csv_file, split_type):
        csv_path = os.path.join(base_path, csv_file)
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping.")
            return

        a = pd.read_csv(csv_path)
        
        # Filter out invalid bounding boxes
        valid_boxes = (a['Roi.X2'] > a['Roi.X1']) & (a['Roi.Y2'] > a['Roi.Y1'])
        a = a[valid_boxes].dropna().reset_index(drop=True)
        
        print(f"Processing GTSRB [{split_type}]...")
        for idx, row in tqdm(a.iterrows(), total=len(a)):
            img_path = os.path.join(base_path, row['Path'])
            if not os.path.exists(img_path):
                continue
            
            # Destination: /OUTPUT_DIR/GTSRB///
            save_dir = os.path.join(OUTPUT_DIR, "GTSRB", split_type, str(int(row['ClassId'])))
            os.makedirs(save_dir, exist_ok=True)
            
            with Image.open(img_path) as img:
                crop_box = (row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2'])
                cropped_img = img.crop(crop_box)
                
                filename = os.path.basename(row['Path'])
                save_path = os.path.join(save_dir, f"{idx}_{filename}")
                cropped_img.save(save_path)

    crop_and_save("Train.csv", "train")
    crop_and_save("Test.csv", "test")

if __name__ == "__main__":
    # Adjust paths if your root folders differ
    ctsd_dir = "data/raw_data/CTSD/"
    gtsrb_dir = "data/raw_data/GTSRB/"
    
    if os.path.exists(ctsd_dir):
        process_ctsd(ctsd_dir)
    
    if os.path.exists(gtsrb_dir):
        process_gtsrb(gtsrb_dir)