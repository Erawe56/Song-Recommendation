import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
import joblib

# Load saved models and preprocessing objects
cosine_sim = joblib.load('cosine_similarity_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
scaler = joblib.load('scaler.pkl')
data = pd.read_csv("D:/music/songs.csv")  # Update path as needed

# Re-fit vectorizer and scaler on entire dataset
vectorizer.fit(data['Lyrics'])
input_features = vectorizer.transform(data['Lyrics'])
scaler.fit(input_features)

# Function to recommend songs based on input
def recommend_songs(input_text, data, cosine_sim, vectorizer, scaler, top_n=5):
    # Preprocess input text (assuming it's either lyrics, artist name, or song name)
    input_features = vectorizer.transform([input_text])

    # Ensure scaler is fitted and then transform input features
    scaler.fit(input_features)
    input_features = scaler.transform(input_features)

    # Reduce dimensionality of input_features to match cosine_sim
    input_features = input_features[:, :cosine_sim.shape[0]]  # Adjust to match cosine_sim's number of features

    # Calculate similarity with all songs
    sim_scores = cosine_similarity(input_features, cosine_sim)

    # Get indices of top-n most similar songs
    sim_scores = sim_scores.flatten()
    top_indices = sim_scores.argsort()[::-1][:top_n]

    # Return recommended songs with artist name
    recommended_songs = data.iloc[top_indices]['Name'] + ' - ' + data.iloc[top_indices]['Artist']
    return recommended_songs

# Streamlit app layout
st.title('Music Recommendation System')

# Input field for user to enter lyrics, artist name, or song name
user_input = st.text_area('Enter lyrics, artist name, or song name:')

if st.button('Recommend'):
    if user_input.strip() == '':
        st.error('Please enter some text.')
    else:
        try:
            # Get recommendations
            recommendations = recommend_songs(user_input, data, cosine_sim, vectorizer, scaler)
            
            # Display recommendations
            st.subheader('Recommended Songs:')
            for song in recommendations:
                st.write(song)
        except ValueError as e:
            st.error(str(e))