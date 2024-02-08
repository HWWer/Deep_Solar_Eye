from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import os
from datetime import datetime
import pandas as pd
import numpy as np


def get_numerical_data() -> pd.DataFrame:
    """
    Preprocesses images from file and returns metadata in a dataframe.
    Always processes and returns the full dataset without any time-based filtering.

    Returns:
    - pd.DataFrame: Metadata Dataframe with only seconds of the day, percentage loss, and irradiance level.
    """

    folder_path = "../raw_data/PanelImages"
    metadata = []  # Initialise an empty list to collect metadata

    # Iterate through files in the specified folder path
    for filename in os.listdir(folder_path):
        if not filename.endswith(".jpg"):
            continue  # Skip files that are not JPG images

        # Split filename to extract metadata
        split_name = filename.split('_')
        hour = int(split_name[4])
        minute = int(split_name[6])
        second = int(split_name[8])
        age_loss = split_name[11]
        irradiance_level = split_name[13][:-4]  # Remove file extension

        # Calculate seconds of the day
        seconds_of_day = hour * 3600 + minute * 60 + second

        # Append extracted information to the metadata list
        filename_info = [seconds_of_day, age_loss, irradiance_level]
        metadata.append(filename_info)

    # Create a DataFrame from the metadata list
    df = pd.DataFrame(metadata, columns=['Seconds of Day', 'Percentage Loss', 'Irradiance Level'])

    # Specify column data types
    df = df.astype({'Seconds of Day': int, 'Percentage Loss': float, 'Irradiance Level': float})

    return df


def load_and_process_image(file_path):
    """
    Loads and preprocesses a single image file.

    Parameters:
    - file_path: str, the path to the image file.

    Returns:
    - img: tf.Tensor, the preprocessed image tensor.
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = preprocess_input(img)
    return img

def load_tensor(df, batch_size):
    """
    Prepares datasets for training, including image preprocessing and data batching.

    Parameters:
    - df: pandas.DataFrame, containing the necessary data columns.
    - batch_size: int, the number of samples per batch in the dataset.

    Returns:
    - all_ds: tf.data.Dataset, a dataset ready for training.
    """
    path_imgs = "../raw_data/PanelImages/*.jpg"  # Make sure this matches your file patterns
    images = tf.data.Dataset.list_files(path_imgs, shuffle=False)

    # Correctly map the load_and_process_image function to each image file path
    images_ds = images.map(load_and_process_image).batch(batch_size)

    # Prepare additional data and target datasets, then batch them
    df_ds = tf.data.Dataset.from_tensor_slices(df[['Seconds of Day', 'Irradiance Level']].values.astype(np.float32)).batch(batch_size)
    y_ds = tf.data.Dataset.from_tensor_slices(df[['Percentage Loss']].values.astype(np.float32)).batch(batch_size)

    # Combine datasets into a final dataset for training
    x_ds = tf.data.Dataset.zip((images_ds, df_ds))
    all_ds = tf.data.Dataset.zip((x_ds, y_ds))

    return all_ds
