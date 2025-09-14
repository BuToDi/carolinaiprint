import streamlit as st
import pandas as pd
from keras.models import load_model
from PIL import Image
import numpy as 
import tensorflow as tf


st.title('WildAIPrint')


uploaded_file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

model = load_model('model.keras')
