import os
from PIL import Image

def preprocess_images(base_folder):
    # Iterate through each class folder inside the dataset (dysgraphic, non_dysgraphic)
    for class_name in ["dysgraphic", "non_dysgraphic"]:
        folder = os.path.join(base_folder, class_name)
        print("Processing folder:", folder)

        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")

        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)

            # Just a safety check to ignore directories
            if os.path.isfile(file_path):
                try:
                    img = Image.open(file_path).convert("L")  # Convert to grayscale
                    img = img.resize((128, 128))
                    img.save(file_path)
                except Exception as e:
                    print(f"Could not process file {file_path}: {e}")
