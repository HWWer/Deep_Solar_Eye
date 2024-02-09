from fastapi import FastAPI, UploadFile, File
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
#from DeepSolarEye.dl_logic.predict import predict_loss

app = FastAPI()

#app.state.model=




@app.get("/")
def root():
    return {
    'API': 'working'
    }


@app.post("/predict")
async def receive_image(img: UploadFile=File(...)):
   ### Receiving and decoding the image

    img = await img.read()

   # model=app.state.model()

    #prediction = predict_loss(model, img)
    return {'power_loss':img}
