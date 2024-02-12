import streamlit as st
import requests
from io import BytesIO
from PIL import Image



#API url
#url=os.getenv('API_URL')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg"])


if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make a POST request to FastAPI endpoint
    response = requests.post("http://127.0.0.1:8000/predict", files={"image": uploaded_file})
    #response = requests.get(url+ "/")


    if response.status_code == 200:
        st.write("Prediction result:", response.json())
    else:
        st.write("Failed to get prediction result")
