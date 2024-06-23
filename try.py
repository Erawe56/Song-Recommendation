import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("D:/music/songs.csv")

# Data cleaning
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Feature extraction: TF-IDF for lyrics
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Lyrics'].fillna(''))

# Combine TF-IDF with popularity
popularity = df['Popularity'].values.reshape(-1, 1)
features = pd.concat([pd.DataFrame(tfidf_matrix.toarray()), pd.DataFrame(popularity)], axis=1)

# Calculate cosine similarity
cosine_sim = cosine_similarity(features, features)

# Recommendation function
def recommend_song(input_text, cosine_sim=cosine_sim, df=df, tfidf=tfidf):
    # Transform input text using the same TF-IDF vectorizer
    input_vector = tfidf.transform([input_text])

    # Calculate similarity with all songs
    sim_scores = cosine_similarity(input_vector, tfidf_matrix)
    sim_scores = sim_scores.flatten()

    # Get indices of top 10 most similar songs
    song_indices = sim_scores.argsort()[-10:][::-1]

    # Return DataFrame with recommended songs
    recommended_songs = df.iloc[song_indices][['Name', 'Artist', 'Album']]
    return recommended_songs

# Streamlit app
def main():
    st.title('Song Recommendation System')

    # Input box for user to enter song name, artist name, or lyrics
    user_input = st.text_input('Enter song name, artist name, or lyrics:', '')

    if st.button('Recommend'):
        if user_input.strip() == '':
            st.write('Please enter a valid input.')
        else:
            recommendations = recommend_song(user_input)
            if recommendations.empty:
                st.write('No recommendations found. Please try a different query.')
            else:
                st.subheader('Recommended Songs:')
                st.write(recommendations)
                
if __name__ == '__main__':
    main()