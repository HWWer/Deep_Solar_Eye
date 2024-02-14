import streamlit as st
import requests
from io import StringIO, BytesIO
from PIL import Image
import ipdb
import base64


#API url
#url=os.getenv('API_URL')


#App title/ description
st.header(':mostly_sunny: Solar Panel Power Loss Estimator :mostly_sunny:')


st.markdown('####')

uploaded_file = st.file_uploader("Select you solar panel image to be analyzed...	:frame_with_picture:", type=["jpg"])




if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    #ipdb.set_trace()
    # Make a POST request to FastAPI endpoint
    # Convert image to byte array

    # Convert image to byte array
    byte_arr = BytesIO()
    image.save(byte_arr, format='JPEG')
    byte_arr = byte_arr.getvalue()

    # Prepare data to send to FastAPI endpoint
    file = {"file": (uploaded_file.name, byte_arr, "image/jpeg")}

    response = requests.post("http://127.0.0.1:8000/predict", files=file)
    #response = requests.get(url+ "/")


    if response.status_code == 200:
        st.write("	:crystal_ball:")
        st.write("Our model predicts the power output to be:", response.json())
    else:
        st.write("  :construction:")
        st.write("Failed to get prediction result")
