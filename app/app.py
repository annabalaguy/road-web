import streamlit as st
import numpy as np
from PIL import Image
import cv2
from helpers import image_to_dict, image_from_dict, display_resized_prediction, binarize_predictions
import requests as rq
import json
import base64
import matplotlib.pyplot as plt




st.set_page_config(
    page_title="ANTI-DEFORESTATION",  # => Quick reference - Streamlit
    page_icon="🌳",
    layout="centered",  # wide
    initial_sidebar_state="auto")  # collapsed

st.title('WELCOME TO THE ROADS DETECTOR')
st.header('Online tool to identify areas of deforestation risk')


st.markdown('''
Are there any road in the landscape ?
''')
image_to_predict = st.file_uploader("Please upload your image here to highlight the roads",
                                    type=['jpg', 'png', 'jpeg'])
#prediction = st.file_uploader("upload the mask", type=['jpg', 'png', 'jpeg'])

if image_to_predict:
    image = Image.open(image_to_predict)
    st.image(image)
    #Save a 526x images to be used later in the display
    image_526 = image.resize((526, 526))
    # Changement de la résolution de l'image
    image=image.resize((256,256))
    rgb_im = image.convert('RGB')
    imgArray = np.array(rgb_im)

if st.button("Click to discover the road prediction 🛣"):
    st.write('Roads incoming…')
    #response = requests.get(url, params).json()

    endpoint = 'https://api-road-sfkiqek4vq-ew.a.run.app/predict'
    # Ensure json content type
    headers = {}
    headers['Content-Type'] = 'application/json'
    # Use helpers method to prepare image for request
    request_dict = image_to_dict(imgArray,dtype='float32')
    # Post image data, and get prediction
    res = rq.post(endpoint, json.dumps(request_dict), headers=headers).json()


    #----- test ------
    final_prediction = image_from_dict(res,dtype='float16')
    final_prediction = final_prediction[0]
    final_prediction = final_prediction.reshape(256, 256)
    final_prediction = binarize_predictions(final_prediction)
    final_prediction = display_resized_prediction(final_prediction)

    # imgArray =  np.float32(imgArray)
    # imgArray = display_resized_prediction(imgArray)
    imgArray = np.asarray(image_526, np.uint16)

    final_prediction = np.uint16(final_prediction)
    mask = np.ma.masked_where(final_prediction < 1, np.zeros(final_prediction.shape))

    final_prediction = Image.fromarray(final_prediction)
    final_prediction = final_prediction.convert("RGB")
    final_prediction = np.asarray(final_prediction, np.uint16)

    if res:
        #st.balloons()
        #st.success('Success!')
        #transform from (1, 256, 256, 1) to (256, 256)
        # array = image_from_dict(res,dtype='float16')[0].reshape(256, 256)
        #binarize
        # array = binarize_predictions(array)
        #resize
        # array = display_resized_prediction(array)

        #SUPERPOSITION PREDICTION IMAGE
        st.header("Find the roads detected below")
        fig, ax = plt.subplots()
        ax.imshow(imgArray)
        ax.imshow(mask, cmap='spring')
        ax.axis('off')
        st.pyplot(fig)
        #MASQUE
        st.header("Below is the prediction only")
        st.image(final_prediction)
