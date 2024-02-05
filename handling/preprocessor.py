import pandas as pd
import numpy as np
import regex as re
import os
import cv2
from datetime import datetime

def preprocess_data(size=('full', 'noon')):
    """
    Preprocesses images from file. Returns metadata in a dataframe, and a np array of image data.

    Takes keywords ['full': entire dataset, 'noon': all images at 12pm]
    """

    def process_image(image):


    folder_path = "raw_data/PanelImages"
    image_data = []  # List to store image data
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

        # Convert the image to a numpy array (tensor)
        # OpenCV loads images in BGR format by default, this convert to RGB
        image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_array.append(image_array)

        filename_info.append(image_array)
        image_data.append(filename_info)


        # Convert the list of tuples to a pandas DataFrame
    df = pd.DataFrame(image_data, columns=['Month', 'Day', 'Date', 'Hour', 'Minute', 'Second', 'Year',
                                        'Datetime', 'Percent Age Loss', 'Irradiance Level'])

    df = df.astype({'Month': str, 'Day': str, 'Date': int, 'Hour': int, 'Minute': int, 'Second': int, 'Year': int,
                                       'Datetime': 'datetime64[ns]', 'Percentage Loss': float, 'Irradiance Level': float})

    return df, image_data
