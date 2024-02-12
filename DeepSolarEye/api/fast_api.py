from fastapi import FastAPI, UploadFile, File
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import sys
sys.path.append('..')
from DeepSolarEye.dl_logic.preprocess_predict import preprocess_predict_loss
from DeepSolarEye.dl_logic.model import regression_ResNet
import os
import ipdb

app = FastAPI()

model_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'model_weights/full_model_adam.h5')
model = regression_ResNet(model_name='ResNet50', input_shape=(224, 224, 3),input_time_irradiance=(2,), num_units=512, pretrained=True)
model.load_weights(model_path)
app.state.model=model



@app.get("/")
def root():
    return {
    'API': 'working'
    }


@app.post("/predict")
async def receive_image(img: UploadFile=File(...)):
   ### Receiving and decoding the image
    filename=img.filename
    img = await img.read()
    x = preprocess_predict_loss(img, filename)

    prediction = app.state.model.predict(x)[0][0]
    return {'power_loss':str(prediction)
            }
