import os
import zipfile

# Build paths in the current directory
current_dir = os.getcwd()
downloads_folder = os.path.join(current_dir, "data")
destination_folder = os.path.join(current_dir, "extracted")

# Create the destination directory if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Loop through files in the data folder
for filename in os.listdir(downloads_folder):
    if filename.endswith(".zip"):
        file_path = os.path.join(downloads_folder, filename)
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(destination_folder)
                print(f"Extracted: {filename} to {destination_folder}")
        except zipfile.BadZipFile:
            print(f"Failed to extract {filename}: Not a valid zip file")

print("Extraction completed.")
