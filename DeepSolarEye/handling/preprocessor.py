import pandas as pd
import numpy as np
import os
import cv2
from datetime import datetime
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf

def preprocess_data(size=('full', 'noon', '15_mins')) -> (pd.DataFrame, np.ndarray):
    """
    Preprocesses images from file. Returns metadata in a dataframe, and a np array of image data.
    Use 'size' kwarg to decide what split of the dataset will be processed and returned.
    'full' = c. 45k images
    'noon' = c. 3.7k images
    '15_mins' = c. 1k images

    Returns: Metadata Dataframe, Tensor np.ndarray
    """

    folder_path = "../raw_data/PanelImages"
    image_data = [] # initialise an empty array to stack the images
    metadata = []
    # Regular expression pattern to extract date and intensity values from the filename
    # Regular expression pattern to extract date and intensity values from the filename
    minute_range = np.arange(0, 15, 1)
    # Convert the numpy array to a list of strings
    minute_range_strings = [str(num) for num in minute_range]
    read_count = 0

    # capped at 1000 for now
    for filename in os.listdir(folder_path):

        if not filename.endswith(".jpg"):
            continue

        split_name = filename.split('_')
        hour = split_name[4]
        minute = split_name[6]
        # put in the break
        if size in ['noon', '15_mins'] and hour != '12':
            continue
        if size == '15_mins' and minute not in minute_range_strings:
            continue
        read_count += 1
        weekday = split_name[1]
        month = split_name[2]
        day = split_name[3]
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

    print(f'loaded {read_count} images')
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

def time_encoder(df: pd.DataFrame, hour_col, minute_col, second_col):
    # Apply cyclical encoding for hour column
    df[hour_col + '_sin'] = np.sin(2 * np.pi * df[hour_col] / 24)
    df[hour_col + '_cos'] = np.cos(2 * np.pi * df[hour_col] / 24)

    # Apply cyclical encoding for minute column
    df[minute_col + '_sin'] = np.sin(2 * np.pi * df[minute_col] / 60)
    df[minute_col + '_cos'] = np.cos(2 * np.pi * df[minute_col] / 60)

    # Apply cyclical encoding for second column
    df[second_col + '_sin'] = np.sin(2 * np.pi * df[second_col] / 60)
    df[second_col + '_cos'] = np.cos(2 * np.pi * df[second_col] / 60)



    # Return dataframe with sin and cos values
    return df

def preprocess_img(img):
    '''Preprocess the image from user and transforms it into a Dataset for model input'''
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = preprocess_input(img)
    img = tf.data.Dataset(img)
    images_ds = tf.data.Dataset.from_tensor_slices(img)
    return images_ds
