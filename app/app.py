import streamlit as st
import numpy as np
from PIL import Image
from helpers import image_to_dict, image_from_dict
import requests as rq
import json
import cv2

def binarize_predictions(pred, threshold=0.5):
    #Binarize the prediction based on if above or below a given threshold (default = 0.5)
    dimension = pred.shape[0]
    new_pred = []
    #vectorize
    pred = list(pred.reshape(dimension*dimension))
    #binarize the vector
    for pixel in pred:
        if pixel > threshold:
            new_pred.append(255)
        else:
            new_pred.append(0)

    new_pred = np.array(new_pred).reshape(dimension, dimension)
    return new_pred


st.set_page_config(
    page_title="ANTI-DEFORESTATION",  # => Quick reference - Streamlit
    page_icon="🌳",
    layout="centered",  # wid
    initial_sidebar_state="auto")  # collapsed

st.title('WELCOME TO THE ROADS DETECTOR')
st.header('Serious tool to prevent risky regions from deforestation')
import base64

st.markdown('''
Are there any road in the landscape ?
''')
image_to_predict = st.file_uploader("Please upload your image here to highlight the roads",
                                    type=['jpg', 'png', 'jpeg'])
#prediction = st.file_uploader("upload the mask", type=['jpg', 'png', 'jpeg'])

if image_to_predict:
    image = Image.open(image_to_predict)
    st.image(image)
    # Changement de la résolution de l'image
    image=image.resize((256,256))
    rgb_im = image.convert('RGB')
    imgArray = np.array(rgb_im)
    st.write(imgArray.shape)

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
    imgArray =  np.float32(imgArray)
    final_prediction = np.uint16(final_prediction)
    final_prediction = Image.fromarray(final_prediction)
    final_prediction = final_prediction.convert("RGB")
    final_prediction = np.asarray(final_prediction, np.uint16)
    imgArray = np.asarray(imgArray, np.uint16)
    #overlay = cv2.imdecode(final_prediction, 1)
    #background = cv2.imdecode(imgArray, 1)
    added_image = cv2.addWeighted(imgArray, 0.4, final_prediction, 0.1, 0)
    # background = np.asarray(bytearray(image_to_predict.read()), dtype=np.uint8)
    # background = cv2.imdecode(background, 1)
    # # Now do something with the image! For example, let's display it:
    #added_image = cv2.addWeighted(background, 0.4, final_prediction, 0.1, 0)


    if res:
        st.balloons()
        st.success('Success!')
        st.write("Here are the real roads seen in the landscape")
        st.image(added_image)
        #st.image(image_from_dict(res,dtype='float16'))
