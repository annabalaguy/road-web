import streamlit as st
import numpy as np
from PIL import Image
from helpers import image_to_dict, image_from_dict
import requests as rq
import json




st.set_page_config(
    page_title="ANTI-DEFORESTATION",  # => Quick reference - Streamlit
    page_icon="🌳",
    layout="centered",  # wide
    initial_sidebar_state="auto")  # collapsed

st.title('WELCOME TO THE ROADS DETECTOR')
st.header('Serious tool to prevent risky regions from deforestation')


st.markdown('''
Is there any road in the landscape ?
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


    if res:
        st.balloons()
        st.success('This is a success!')
        st.write("Here are the real roads seen in the landscape")
        st.image(image_from_dict(res,dtype='float16'))
