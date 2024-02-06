import pandas as pd
import numpy as np
import regex as re
import os
import cv2
from datetime import datetime

def preprocess_data(size=('full', 'noon')) -> (pd.DataFrame, np.ndarray):
    """
    Preprocesses images from file. Returns metadata in a dataframe, and a np array of image data.
    Use 'size' kwarg to decide what split of the dataset will be processed and returned.

    Returns: Metadata Dataframe,
    """

    folder_path = "../raw_data/PanelImages"
    image_data = [] # initialise an empty array to stack the images
    metadata = []
    # Regular expression pattern to extract date and intensity values from the filename

    # capped at 1000 for now
    for filename in os.listdir(folder_path):
        split_name = filename.split('_')
        hour = split_name[4]
        if size == 'noon' and hour != '12':
            continue
        weekday = split_name[1]
        month = split_name[2]
        day = split_name[3]
        minute = split_name[6]
        second = split_name[8]
        year = split_name[9]
        datetime_obj = datetime.strptime(f"{month} {day} {year} {hour}:{minute}:{second}", "%b %d %Y %H:%M:%S")
        age_loss = split_name[11]
        irradiance_level = split_name[13][:-4]

        # append metadata to list
        filename_info = [month, weekday, day, hour, minute, second, year, datetime_obj, age_loss, irradiance_level]

        metadata.append(filename_info)

        file_path = os.path.join(folder_path, filename)

        # Load the image using OpenCV
        image = cv2.imread(file_path)

        # Resize the image to 224x224 using bilinear interpolation - OPTION to save the resized images so this never is done again!
        resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Convert the image to a numpy array (tensor)
        # OpenCV loads images in BGR format by default, this convert to RGB
        image_array = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        image_data.append(image_array)

    # Convert the list of tuples to a pandas DataFrame
    df = pd.DataFrame(metadata, columns=['Month', 'Day', 'Date', 'Hour', 'Minute', 'Second', 'Year',
                                        'Datetime', 'Percentage Loss', 'Irradiance Level'])

    df = df.astype({'Month': str, 'Day': str, 'Date': int, 'Hour': int, 'Minute': int, 'Second': int, 'Year': int,
                                       'Datetime': 'datetime64[ns]', 'Percentage Loss': float, 'Irradiance Level': float})

    # convert image data to numpy arrays
    image_data = np.array(image_data)
    # normalize the image data
    image_data = image_data / 255.0

    # return metadata and normalized image data
    return df, image_data
