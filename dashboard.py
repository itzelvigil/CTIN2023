import streamlit as st
import numpy as np 
import pandas as pd 

option = st.sidebar.selectbox('select option',['General data','Explotary Data Analysis by material','Timeseries Analysis','Forecasting','Materials','Purchase orders','Customer orders'])
st.title('CTIN 2023 Taller de ciencia de datos')