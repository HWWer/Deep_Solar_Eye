import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import cv2
import os
import ipdb
from DeepSolarEye.handling.preprocessor import preprocess_img
from DeepSolarEye.dl_logic.model import regression_ResNet

#Create instance
model = regression_ResNet()

#Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#load weights

model_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'model_weights/first_model.h5')

model = model.load_weights(model_path)

#laod image
image_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
,'raw_data/PanelImages/solar_Fri_Jun_16_6__0__25_2017_L_0.0901960784314_I_0.003.jpg')

img= cv2.imread(image_path)



def preprocess_predict_loss(img,filename):

    images_ds = preprocess_img(img)

    ipdb.set_trace()
    if filename.startswith('solar_'):

        split_name = filename.split('_')
        hour = int(split_name[4])
        minute = int(split_name[6])
        second = int(split_name[8])
        age_loss = split_name[11]
        irradiance_level = split_name[13][:-4]  # Remove file extension

        # Calculate seconds of the day
        seconds_of_day = hour * 3600 + minute * 60 + second

        # Append extracted information to the metadata list
        filename_info = [[seconds_of_day, irradiance_level]]


        # Create a DataFrame from the metadata list
        df = pd.DataFrame(filename_info, columns=['Seconds of Day', 'Irradiance Level'])

        # Specify column data types
        df = df.astype({'Seconds of Day': int, 'Irradiance Level': float})
        # Apply Min-Max scaling to Seconds of the Day and Irradiance Level



    else:
        #Fixed input for seconds/time and irradiance level. Set to max values
        fixed_input = {'Seconds of Day': 43200,
                'Irradiance Level': 1}
        df=pd.DataFrame([fixed_input])


    #Scaling num features
    scaler = MinMaxScaler()
    df[['Seconds of Day', 'Irradiance Level']] = scaler.transform(df[['Seconds of Day', 'Irradiance Level']])
    #Add index to DataFrame
    df['index'] = range(len(df))
    df.set_index('index', inplace=True)


    df_ds = tf.data.Dataset.from_tensor_slices(df[['Seconds of Day', 'Irradiance Level']].values.astype(np.float32))

    #combine
    x_ds = tf.data.Dataset.zip((images_ds, df_ds))


    #loss_prediction= model.predict(x_ds)

    return x_ds
