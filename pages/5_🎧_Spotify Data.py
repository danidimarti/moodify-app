import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

from matplotlib import pyplot as plt

# Define the CSS style for the container
container_style = """
    .text-container {
        background-color: #212121;
        border: 1px solid #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        color: #D3D3D3;
        font-size: 1.1rem;
        font-family: 'Open Sans', sans-serif;
        
    }
"""

bullet_style = """
    .bullet {
        background-color: #212121;
        border-radius: 10px;
        padding: 20px;
        color: #D3D3D3;
        font-size: 1.3rem;
        font-family: 'Open Sans', sans-serif;
        
    }
"""

with open('style.css') as f: 
    st.markdown(f'<style>{f.read()} </style>', unsafe_allow_html=True)

    
st.markdown("<h1 style='color:#9FE1B4; font-family: serif;'>Spotify Data</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='color:#9FE1B4;'>Emotion labeled dataset from Spotify Api</h3>", unsafe_allow_html=True)

st.write(
    "<div style='font-size:1.2rem'>"
    "The Spotify API provides developers with access to a massive library of songs and metadata. I utilized this API to extract songs from playlists that corresponded to the four emotion categories that I selected when creating the CNN model: angry, happy, sad, and neutral.</div>",
    unsafe_allow_html=True
)

######### ------ SPOTIFY SETTINGS  ------------- ########

token = "<div style='font-size:1.2rem;'><ol type='1'><li>Get auth for App development (CLIENT_ID, CLIENT_SECRET) </li><li>Get auth for token for the queries.</li></ol></div>"

st.markdown(f'<div class="bullet-points"><style>{bullet_style}</style><div style="color:#FFDAB9; font-family: serif; font-size:1.7rem">Spotify Token</div>{token}</div><br>', unsafe_allow_html=True)


######----- GET MOOD PLAYLISTS ------ ######

getplaylists = """
<div style='font-size:1.2rem;'>
<ol>
  <li>Find mood playlists for each of the Model sentiments used</li>
  <li>Extract playlist id</li>
  <li>Create function to extract playlist data from Api based on the structure of the response dictionary.</li>
  <li>Create emotion column for each dictionary</li>
  <li>Append information to list</li>
  <li>Create df</li>
  <li>Save csv file</li>
</ol>
</div>
"""

st.markdown(f'<div class="bullet-points"><style>{bullet_style}</style><div style="color:#FFDAB9; font-family: serif; font-size:1.7rem">Steps:</div>{getplaylists}</div><br>', unsafe_allow_html=True)


##### ----- VISUALIZE THE DF ------ ####
st.write(
    "<div style='font-size:1.2rem;'>"
    "I used a total of 15 playlists, from the app's mood feature. This feature allows users to browser playlists and songs that are curated based  on different emotions. The resulting df looks like this:</div><br>",
    unsafe_allow_html=True
)

st.image('imgs/Spotify_sample.png', caption='Fer2013_Emotion Count', use_column_width=True)

st.write(
    "<div style='font-size:1.2rem;'>"
    "<br>Let\'s visualize the number of songs per emotion we have:</div><br>",
    unsafe_allow_html=True
)

st.image('imgs/Spotify_label-count.png', caption='Fer2013_Emotion Count', use_column_width=True)



##### ---------- SCALLING UP THE RECOMMENDATION SYSTEM --------- ####

st.markdown("<br><h2  style='color:#9FE1B4; font-family: serif;'>Scaling Spotify Recommendation system:</h2>", unsafe_allow_html=True)


scalling = "<div style='font-size:1.2rem;'>To improve the scalability of our Spotify recommendation system, we need to move away from using a limited playlist csv file and find songs directly within the Spotify platform. One of the most used psychological models of the factor structure of emotions is the Valence-Arousal Plane. It is a two dimensional model reducing emotions to their arousal (energy) and valence components. Although this model is not perfect, it can certainly help us give the next step into our mood application.<a href='https://developer.spotify.com/discover/#recommendations'> Spotify uses various features, such as acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, valence, and tempo to classify tracks.</a> We will use the song features as controls to offer mood specific songs. All we need is something that the app can use as a reference. <br> To do this, I retrieved a dataset from <a href='https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset'>Kaggle</a> containing 171k songs with all their specific feature details.</div>"


st.markdown(f'<div class="container_style"><style>{container_style}</style>{scalling}</div><br>', unsafe_allow_html=True)

st.image('imgs/df_song-features.png', caption='Spotify song feature sample', use_column_width=True)


definitonspotify = '<b>Valence:</b><br> Describes the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). <br> <br> <b> Energy: </b> </br> Numerical, Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.'

st.markdown(f'<div class="text-container"><style>{container_style}</style><div style="color:#FFDAB9; font-family: serif; font-size:1.5rem">For curiosity, this is how Spotify define these two variables inside the API documentation:</div>{definitonspotify}</div><br><br>', unsafe_allow_html=True)


##### ------ ELBOW GRAPH AND SUMMARY ----- ##### 
elbow = "<div style='font-size:1.2rem;'>I used the K model and the elbow method to identify the number of clusters. We then looked at the mean values of the features in each cluster to understand the characteristics of the songs in each cluster. <br> I choose the K-means because it is a popular clustering algorithm because it's simple, fast, and relatively easy to understand. </div>"


st.markdown(f'<div class="container_style"><style>{container_style}</style>{elbow}</div><br>', unsafe_allow_html=True)

st.image('imgs/elbow_method.png', caption='Elbow Method for clustering definition', use_column_width=True)


elbowsummary = 'I noticed that around cluster 4 the lines changes deaccelerates dramatically. Luckily this is also the number of emotions I have mapped on the facial recognition model so I decided to keep this number.'

st.markdown(f'<div class="text-container"><style>{container_style}</style><div style="color:#FFDAB9; font-family: serif; font-size:1.5rem">Elbow Method Summary:</div>{elbowsummary}</div><br><br>', unsafe_allow_html=True)


###### ------ VALENCE AND AROUSAL PLANE ------ #####

st.image('imgs/The Valence-Arousal Plane and the Locations of Several Emotions-Moods on it (adapted from Russel, 1980).png', caption='Quadrant Plot Energy vs Valence', use_column_width=True)

plane = "<div style='font-size:1.2rem;'>The Valence-Arousal Plane is a commonly used way to describe and understand emotions using two dimensions: valence and arousal. <br> Just like shown in the Spotify definitions the valence refers to how positive or negative an emotion is and arousal refers to how energized someone feels (anxious vs calm). <br>I have used this plane to plot and understand different emotions.  <br> <br></div>"


st.markdown(f'<div class="container_style"><style>{container_style}</style>{plane}</div><br>', unsafe_allow_html=True)



###### ----- EMOTIONS CLUSTERS ----- ##### 


image = Image.open('imgs/valence_energy.png')

# Resize the image
new_size = (int(image.size[0] * 0.8), int(image.size[1] * 0.8))
resized_image = image.resize(new_size)

# Display the resized image
st.image(resized_image, caption='Valence and energy per cluster')


clustermeans = "<div style='font-size:1.2rem;'>Based on the Spotify definitions and the Valance-arousal plane, I mapped each cluster to an emotion. For example, a cluster with high mean values for danceability, valence, and energy was mapped to 'happy' songs.</div>"


st.markdown(f'<div class="container_style"><style>{container_style}</style>{clustermeans}</div><br>', unsafe_allow_html=True)


clusters = '<b>Cluster 0 - Neutral:</b><br>Valence: 0.655 (high)<br>Energy: 0.325 (neutral)<br>Explanation: Cluster 1 has neutral energy and neutral mean valence, indicating that the songs in this cluster could be classified as Neutral.<br><br><b>Cluster 1 - Angry:</b><br>Valence: 0.383 (neutral)<br>Energy: 0.741 (high)<br>Explanation: Cluster 0 has a high energy level and neutral valence, which suggests that the songs in this cluster could be classified as angry.<br><br><b>Cluster 2 - Sad:</b><br>Valence: 0.233 (low)<br>Energy: 0.213 (low)<br>Explanation: Cluster 2 has the lowest values for valence and energy which suggests that the songs in this cluster could be classified as sad.<br><br><b>Cluster 3 - Happy: </b> <br>Valence: 0.807 (high)<br>Energy: 0.714 (high)<br>Explanation: Cluster 3 has high values for valence and energy indicating that the songs in this cluster could be classified as happy.<br><br>'

st.markdown(f'<div class="text-container"><style>{container_style}</style><div style="color:#FFDAB9; font-family: serif; font-size:1.5rem">Clusters</div>{clusters}</div><br><br>', unsafe_allow_html=True)


##### ----- Sample Song Plot ----- #####
plotsong = "<div style='font-size:1.2rem;'>I plotted a sample song from each cluster on the valence-energy plane to see how it fits into each quadrant. I  used the following colors to represent each emotion: green for happy, cyan for neutral, red for angry, and blue for sad. The frequency of each cluster was also plotted in a bar chart.</div>"


st.markdown(f'<div class="container_style"><style>{container_style}</style>{plotsong}</div><br>', unsafe_allow_html=True)

st.image('imgs/emotion-quadrant_plot.png', caption='Sample song per label', use_column_width=True)


##### ----- Label count bar plot for the machine leanring model ---- #####

labelcount = "<div style='font-size:1.2rem;'>Finally let\'s do a lable count of the new labels for better visualization:</div>"


st.markdown(f'<div class="container_style"><style>{container_style}</style>{labelcount}</div><br>', unsafe_allow_html=True)


st.image('imgs/Spotify_label-count.png', caption='machine learning label counter plot', use_column_width=True)









