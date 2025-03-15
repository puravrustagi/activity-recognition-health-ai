import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# pip install kaggle 
# pip install kagglehub
#run this above command if the kaggle is not installed on your machine


# In[3]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("uciml/human-activity-recognition-with-smartphones")

print("Path to dataset files:", path)
st.write("Welcome to CapStone Project on human-activity-recognition-with-smartphones")
st.write("Imporpting the Dataset human-activity-recognition-with-smartphones")
