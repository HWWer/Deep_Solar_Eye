import os
import random
import shutil
from tqdm import tqdm
def train_test_val_split(train_ratio=0.64, test_ratio=0.2, val_ratio=0.16):
    """
    Function to randomly split data into three dirs 'train', 'test', 'val'.
    If any 'train, test, val' folders already exist, they are overwritten.
    Default splits are set to 64% train, 20% test, and 16% val of total files.
    """


    current_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    #input_folder="/raw_data/PanelImages"
    #output_folder = "/raw_data/"
    input_folder = os.path.join(current_folder, "raw_data/PanelImages")
    output_folder = os.path.join(current_folder, "raw_data/")

    # Check if the input folder exists
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder '{input_folder}' not found.")

    # Create output folders if they don't exist
    for folder in ['train_data', 'test_data', 'val_data']:
        # if folders exist
        folder_path = os.path.join(output_folder, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)  # Delete existing folder

        os.makedirs(os.path.join(output_folder, folder), exist_ok=False)

    # List files in the input folder and filter out files that do not end with .jpg
    image_files = [file for file in os.listdir(input_folder) if file.lower().endswith('.jpg')]

    print(len(image_files))
    # Shuffle the list of image files
    random.shuffle(image_files)

    percentage = train_ratio + test_ratio + val_ratio

    num_files = len(image_files)
    num_train = int(train_ratio * num_files)
    num_test = int(test_ratio * num_files)
    num_val = int(val_ratio * num_files)

    image_files = image_files[:int(len(image_files) * percentage)]
    print(len(image_files))
    # Calculate the number of files for each split

    # Split the files into train, test, and val sets by slicing list of filenames
    train_files = image_files[:num_train]
    test_files = image_files[num_train:num_train+num_test]
    val_files = image_files[num_train+num_test:]

    # Copy files to their respective folders
    for files, folder in tqdm(zip([train_files, test_files, val_files], ['train_data', 'test_data', 'val_data'])):
        for file in files:
            shutil.copy(os.path.join(input_folder, file), os.path.join(output_folder, folder, file))

    print(f'train size: {num_train}\ntest_size: {num_test}\nval_size: {num_val}')
