import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
print("Path to dataset files:", path)

# Current directory
current_dir = os.getcwd()

# Check if there's a Data folder inside the downloaded path
data_source = os.path.join(path, "Data")
if os.path.exists(data_source):
    # If Data folder exists in downloaded files, copy its contents to current directory's Data folder
    data_dir = os.path.join(current_dir, "Data")
    
    # Remove existing Data folder if it exists
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print(f"Removed existing Data folder: {data_dir}")
    
    # Copy the Data folder from downloaded path
    shutil.copytree(data_source, data_dir)
    print(f"Dataset copied to: {data_dir}")
    
else:
    # If no Data folder, copy all contents to Data folder
    data_dir = os.path.join(current_dir, "Data")
    
    # Remove existing Data folder if it exists
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print(f"Removed existing Data folder: {data_dir}")
    
    # Create Data folder and copy contents
    os.makedirs(data_dir, exist_ok=True)
    
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(data_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    
    print(f"Dataset copied to: {data_dir}")

# Check what's inside
print(f"\nContents of {data_dir}:")
for item in os.listdir(data_dir):
    item_path = os.path.join(data_dir, item)
    if os.path.isdir(item_path):
        print(f"  Folder: {item}/")
    else:
        print(f"  File: {item}")