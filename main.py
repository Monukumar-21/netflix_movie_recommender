import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation Engine", page_icon="🎬", layout="wide")

@st.cache_resource
def load_assets():
    df = joblib.load('movies_metadata.pkl')
    prob_matrix = joblib.load('cluster_probabilities.pkl')
    model = joblib.load('gmm_model.pkl') 
    return df, prob_matrix, model

try:
    df, prob_matrix, gmm = load_assets()
except FileNotFoundError:
    st.error("Deployment assets (.pkl files) not found. Please ensure they are in the project folder.")
    st.stop()

def get_advanced_recommendations(movie_title, df, prob_matrix, top_n=5):
    movie_idx = df[df['Title'] == movie_title].index[0]
    movie_vector = prob_matrix[movie_idx].reshape(1, -1)
    
    # Fast cosine similarity matrix multiplication
    similarities = cosine_similarity(movie_vector, prob_matrix)[0]
    
    temp_df = df.copy()
    temp_df['Similarity_Score'] = similarities
    temp_df = temp_df.drop(movie_idx)
    
    recommendations = temp_df.sort_values(
        by=['Similarity_Score', 'Popularity', 'Vote_Average'], 
        ascending=[False, False, False]
    )
    return recommendations.head(top_n)

st.title("🎬 GMM Soft-Clustering Movie Recommender")
st.write("Enter a movie you like, and the engine will analyze its structural cluster signatures to find similar films.")

all_movies = sorted(df['Title'].unique())
selected_movie = st.selectbox("Search or select a movie:", all_movies)

if st.button("Generate Recommendations"):
    with st.spinner("Analyzing cluster alignment matrices..."):
        results = get_advanced_recommendations(selected_movie, df, prob_matrix, top_n=5)
        
        st.subheader(f"Top Recommendations for '{selected_movie}':")
        
        cols = st.columns(5)
        
        for idx, (_, row) in enumerate(results.iterrows()):
            with cols[idx]:
                if pd.notna(row['Poster_Url']):
                    st.image(row['Poster_Url'], use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/500x750?text=No+Poster", use_container_width=True)
                
                st.markdown(f"**{row['Title']}**")
                st.caption(f"📅 Release: {row['Release_Date']}")
                st.caption(f"🤝 Match Score: {row['Similarity_Score']:.2%}")
                st.caption(f"⭐ Rating: {row['Vote_Average']}/10")