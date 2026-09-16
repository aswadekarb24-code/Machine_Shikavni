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
            
            # Destination: data/processed_data/CTSD/train|test/class_id/
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
            
            # Destination: data/processed_data/GTSRB/train|test/class_id/
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

def process_btsd(base_path):
    """Processes Belgium (BTSD/BelgiumTSC) train/test directories and crops ROIs."""
    
    def crop_and_save(folder_name, split_type):
        split_path = os.path.join(base_path, folder_name)
        
        # Check alternative folder names if nested inside subfolders
        if not os.path.exists(split_path):
            alt_path = os.path.join(base_path, f"BelgiumTSC_{folder_name}")
            if os.path.exists(alt_path):
                split_path = alt_path
            else:
                print(f"Warning: {split_path} not found. Skipping.")
                return

        # Locate all annotation CSV files (e.g. GT-00000.csv) in class directories
        csv_files = []
        for root, _, files in os.walk(split_path):
            for file in files:
                if file.startswith("GT-") and file.endswith(".csv"):
                    csv_files.append(os.path.join(root, file))

        if not csv_files:
            print(f"Warning: No annotation CSV files found in {split_path}.")
            return

        print(f"Processing BTSD [{split_type}]...")
        for csv_path in csv_files:
            dir_path = os.path.dirname(csv_path)
            
            # Belgium TSC CSV files use semicolon separators
            a = pd.read_csv(csv_path, sep=";")
            a.columns = a.columns.str.strip()
            
            # Standardize column naming to match GTSRB format (case-insensitive fixing)
            col_map = {
                'Roi.x1': 'Roi.X1', 'Roi.y1': 'Roi.Y1',
                'Roi.x2': 'Roi.X2', 'Roi.y2': 'Roi.Y2'
            }
            a = a.rename(columns=col_map)
            
            # Filter out invalid bounding boxes
            valid_boxes = (a['Roi.X2'] > a['Roi.X1']) & (a['Roi.Y2'] > a['Roi.Y1'])
            a = a[valid_boxes].dropna().reset_index(drop=True)

            for idx, row in a.iterrows():
                img_path = os.path.join(dir_path, row['Filename'])
                if not os.path.exists(img_path):
                    continue

                class_id = int(row['ClassId'])
                # Destination: data/processed_data/BTSD/train|test/class_id/
                save_dir = os.path.join(OUTPUT_DIR, "BTSD", split_type, str(class_id))
                os.makedirs(save_dir, exist_ok=True)

                with Image.open(img_path) as img:
                    crop_box = (row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2'])
                    cropped_img = img.crop(crop_box)

                    filename = os.path.basename(row['Filename'])
                    filename_no_ext = os.path.splitext(filename)[0]
                    save_path = os.path.join(save_dir, f"{idx}_{filename_no_ext}.png")
                    cropped_img.convert("RGB").save(save_path)

    crop_and_save("Training", "train")
    crop_and_save("Testing", "test")

if __name__ == "__main__":
    ctsd_dir = "data/raw_data/CTSD/"
    gtsrb_dir = "data/raw_data/GTSRB/"
    btsd_dir = "data/raw_data/BTSD/"
    
    if os.path.exists(ctsd_dir):
        process_ctsd(ctsd_dir)
    
    if os.path.exists(gtsrb_dir):
        process_gtsrb(gtsrb_dir)

    if os.path.exists(btsd_dir):
        process_btsd(btsd_dir)