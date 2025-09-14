import streamlit as st
import pandas as pd
from keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf

df = pd.read_csv('dataphoto2.csv')


st.title('WildAIPrint')


model = load_model('model.keras')
class_names = df['Espèce']


uploaded_file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

if uploaded_file is not None :
    img = Image.open(uploaded_file)
    img_array = np.array(img)

 with st.spinner("Prédiction"):
        pred = model.predict(img_array)
        predicted_class = class_names[np.argmax(pred)]
        animal_info = df[df["Espèce"] == predicted_class]
        info = animal_info.iloc[0]
