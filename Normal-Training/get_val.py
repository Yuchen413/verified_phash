import pandas as pd
import shutil
import os
from pathlib import Path

# Path to your CSV file
csv_file = 'coco-val.csv'

# Folder to save the images
save_folder = 'coco100x100_val'
Path(save_folder).mkdir(parents=True, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file, header=None)

# Loop over the image paths in the first column
for index, row in df.iterrows():
    image_path = row[0]  # Assuming the first column has the image paths
    if os.path.exists(image_path):  # Check if the file exists
        # Construct the path to save the image
        save_path = os.path.join(save_folder, os.path.basename(image_path))
        # Copy the file to the new directory
        shutil.copy(image_path, save_path)
        print(f'Copied {os.path.basename(image_path)} to {save_folder}')
    else:
        print(f'File not found: {image_path}')

print("All images processed.")
