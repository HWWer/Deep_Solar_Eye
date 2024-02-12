import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split

#test comment
def process_solar_panel_image(image_path, gaussian_kernel=(3, 3), intensity_threshold=120):
    # Load the image

    solar_panel_img = cv2.imread(image_path)

    # Apply Gaussian filter to the image
    gaussian_blurred = cv2.GaussianBlur(solar_panel_img, gaussian_kernel, 0)

    # Convert the image to gray scale
    gray_img = cv2.cvtColor(gaussian_blurred, cv2.COLOR_BGR2GRAY)

    # Use thresholding to isolate the solar panel
    _, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours to identify the edges of the solar panel
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour corresponds to the edge of the solar panel
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    largest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    # Create a mask for the solar panel
    mask = np.zeros_like(gray_img)
    cv2.drawContours(mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)

    # Classify pixels outside of the solar panel as Class 1
    classified_img = np.zeros_like(gray_img)
    classified_img[mask == 0] = 1  # Class 1

    # Temporary Class 2 for all pixels inside the solar panel
    classified_img[mask == 255] = 2  # Class 2 temporarily

    # Identify the non-dark pixels within the solar panel
    non_dark_pixels = (gray_img > intensity_threshold) & (mask == 255)

    # Classify non-dark pixels as Class 3
    classified_img[non_dark_pixels] = 3  # Class 3

    return classified_img

def clear_or_create_directory(directory_path):
    """
    Clears the directory if it exists, or creates it if it does not.
    """
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path, exist_ok=True)

def split_data(image_paths, train_size=0.6, val_size=0.2, test_size=0.2):
    """
    Split the data into train, validation, and test sets.
    """
    assert train_size + val_size + test_size == 1, "The sizes must sum up to 1."

    train_val_paths, test_paths = train_test_split(image_paths, test_size=test_size)
    train_paths, val_paths = train_test_split(train_val_paths, test_size=val_size / (train_size + val_size))

    return train_paths, val_paths, test_paths

def copy_and_create_masks(image_paths, dataset_type, rcnn_mask_dir, panel_images_dir):
    """
    Copy original images and create masks for a specific dataset type (train, val, test).
    """
    dataset_dir = os.path.join(rcnn_mask_dir, dataset_type)
    images_dir = os.path.join(dataset_dir, "images")
    masks_dir = os.path.join(dataset_dir, "masks")

    # Clear or create directories
    clear_or_create_directory(images_dir)
    clear_or_create_directory(masks_dir)

    for image_path in tqdm(image_paths, desc=f"Processing {dataset_type} images"):
        image_name = os.path.basename(image_path).split('.')[0]  # Get the base name without the extension
        # Process image to get the classified mask
        classified_img = process_solar_panel_image(image_path)
        # Save the mask
        create_masks_for_rcnn(classified_img, image_name, masks_dir)
        # Copy the original image
        shutil.copy(image_path, os.path.join(images_dir, os.path.basename(image_path)))

def create_masks_for_rcnn(classified_img, image_name, rcnn_mask_dir):
    # Create separate masks for each class
    mask_background = np.where(classified_img == 1, 255, 0).astype(np.uint8)
    mask_solar_panel = np.where(classified_img == 2, 255, 0).astype(np.uint8)
    mask_soil = np.where(classified_img == 3, 255, 0).astype(np.uint8)

    # Construct the full path for each mask
    background_path = os.path.join(rcnn_mask_dir, image_name + '_background.png')
    solar_panel_path = os.path.join(rcnn_mask_dir, image_name + '_solar_panel.png')
    soil_path = os.path.join(rcnn_mask_dir, image_name + '_soil.png')

    # Save each mask as an image
    cv2.imwrite(background_path, mask_background)
    cv2.imwrite(solar_panel_path, mask_solar_panel)
    cv2.imwrite(soil_path, mask_soil)

    return background_path, solar_panel_path, soil_path

def batch_process_images(batch_size=50):
    current_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    panel_images_dir = os.path.join(current_folder, "raw_data/PanelImages")
    rcnn_mask_dir = os.path.join(current_folder, "raw_data/RCNN_Masks")
    image_paths = glob.glob(os.path.join(panel_images_dir, "*.jpg"))

    # Split the dataset
    train_paths, val_paths, test_paths = split_data(image_paths)

    # Clear or create the root RCNN_Masks directory
    clear_or_create_directory(rcnn_mask_dir)

    # Process and copy images and masks for each dataset
    for dataset_type, paths in zip(["train", "val", "test"], [train_paths, val_paths, test_paths]):
        copy_and_create_masks(paths, dataset_type, rcnn_mask_dir, panel_images_dir)



if __name__ == "__main__":
    batch_process_images()
