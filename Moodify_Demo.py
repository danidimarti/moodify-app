
import cv2
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import numpy as np
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
import pandas as pd
import time
import os
import statistics
import webbrowser
import detector as dt
import getsong as gs

CLIENT_ID = os.environ.get('CLIENT_ID')
CLIENT_SECRET = os.environ.get('CLIENT_SECRET')

with open('style.css') as f: 
    st.markdown(f'<style>{f.read()} </style>', unsafe_allow_html=True)

st.image('imgs/streamlit-header_temp.png', use_column_width=True)

#df = pd.read_csv('Spotify/data/kaggle_song-features.csv')
df = pd.read_csv('Spotify/data/songs_with_emotions_valence-energy.csv')
seed_song = df.sample()


if st.button('Moodify me', key="center_button"):
    emotion = dt.detect_emotion()
    #emotion_up = emotion.capitalize()

    # Display different message based on the detected emotion
    if emotion == 'happy':
        st.subheader(f"It is a great day to feel {emotion}! I\'ve got just the jam for you.")
    elif emotion == 'sad':
        st.subheader(f"Sorry to see that you are {emotion}. Remember that there's always calm after a storm.")
    elif emotion == 'angry':
        st.subheader(f"Let your {emotion} out. Just try not to break anything.")
    elif emotion == 'neutral':
        st.subheader(f"I see you are feeling {emotion}. Very stoic of you. I dig it")
    
    subset = dt.subsets['sub_' + emotion]
    mood_result = subset.sample()
    link = mood_result['track_link'].iloc[0]
    link_2 = gs.getsong(seed_song)
    # Use st.empty() to create a placeholder for the countdown
    countdown_text = st.empty()
    
    # Start the countdown
    for i in range(3, 0, -1):
        time.sleep(1)
        countdown_text.write(f"Opening song in {i} seconds...")
    # Remove the countdown message after the countdown is complete
    countdown_text.empty()
    
    # Open the link
    webbrowser.open_new_tab(link_2)
 
    
    # Add the footer text at the bottom of the page
st.write(
    "<div style='position: fixed; bottom: 0; width: 100%; background-color: black; color: lightgrey; text-align: center; font-size:10px'>"
    "© 2023 Daniela Demarchi. All Rights Reserved."
    "</div>",
    unsafe_allow_html=True
)