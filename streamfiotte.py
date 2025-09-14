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



# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))