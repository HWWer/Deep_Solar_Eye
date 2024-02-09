import pandas as pd
import numpy as np
import tensorflow as tf


from DeepSolarEye.handling.preprocessor import preprocess_img
from DeepSolarEye.dl_logic.model import regression_ResNet



def predict_loss(model, img):
    images_ds = preprocess_img(img)

    #Fixed input for seconds/time and irradiance level. Set to max values
    fixed_input = {'Seconds of Day': 12,
               'irradiance_level': 1}
    df=pd.DataFrame(fixed_input)
    df_ds = tf.data.Dataset.from_tensor_slices(df[['Hour', 'Irradiance Level']].values.astype(np.float32))

    #combine
    x_ds = tf.data.Dataset.zip((images_ds, df_ds))

    model = regression_ResNet()

    loss_prediction= model.predict(x_ds)

    return loss_prediction
