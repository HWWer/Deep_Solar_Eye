from fastapi import FastAPI, UploadFile, File
from io import BytesIO
import sys
sys.path.append('..')
from deep_solar_prediction import mrcnn_load_inference_model, make_mrcnn_prediction
from skimage.io import imread
import base64


app = FastAPI()

model_path='Solar_mask_rcnn_quarter_data_trained_all.h5'
model = mrcnn_load_inference_model(filepath=model_path)
app.state.model = model

@app.get("/")
def root():
    return {
    'API': 'working'
    }

@app.post("/predict")
async def receive_image(file: UploadFile = File(...)):
   ### Receiving and decoding the image into np array
    contents = await file.read()
    image_file = BytesIO(contents)
    image_array = imread(image_file)

    # make prediction with pre-loaded model
    results = app.state.model.detect([image_array], verbose=1)

    # apply inferred masks to image, returns edited image it byte stream
    inferred_img = make_mrcnn_prediction(image=image_array, results=results)

    # inferred_img is none if no solar panel is predicted above threshold score 0.9
    if inferred_img:
        # Encode the byte stream as a base64 string
        image_base64 = base64.b64encode(inferred_img.getvalue()).decode()

    else:
        image_base64 = None

    # returns None for image if no solar panel detected, marker for is_panel
    return {'inferred_img': image_base64,
            }

## when called, check to see if inferredimg is a string. if so, keep showing existing photo and write pls try again message
