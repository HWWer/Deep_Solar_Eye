from fastapi import FastAPI, UploadFile, File
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import sys
sys.path.append('..')
from DeepSolarEye.dl_logic.predict import predict_loss
from DeepSolarEye.dl_logic.model import regression_ResNet

app = FastAPI()






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
    model=regression_ResNet(model_name='ResNet50', input_shape=(224, 224, 3),input_time_irradiance=(2,), num_units=512, pretrained=True)


    prediction = predict_loss(model,img,filename)
    return {'power_loss':prediction,
            'filename': filename}
