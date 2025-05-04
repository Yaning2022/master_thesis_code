import os
import shutil
import json

# Define source and destination directories
source_folder = "/proj/afraid/users/x_wayan/Data/AFF_NFF_train_color_64/"
destination_folder = "/proj/afraid/users/x_wayan/Data/AFF_NFF_train_color_labels_64"

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# List all files in the source folder
files = [f for f in os.listdir(source_folder) if f.endswith(".png")]

# Create a JSON structure
dataset = {"labels": []}

# Process each file
for file in files:
    parts = file.split("_")
    
    # Extract label
    label = parts[2]  # "NFF" or "AFF"
    label_value = 0 if label == "NFF" else 1
    
    # Add to JSON structure
    dataset["labels"].append([file, label_value])
    
    # Copy image to destination folder
    shutil.copy(os.path.join(source_folder, file), os.path.join(destination_folder, file))

# Save JSON file
json_path = os.path.join(destination_folder, "dataset.json")
with open(json_path, "w") as json_file:
    json.dump(dataset, json_file, indent=4)
