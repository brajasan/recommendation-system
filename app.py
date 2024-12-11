import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import folium
from streamlit_folium import folium_static

def recommend_restaurants(df, restaurant_name, top_n=5):
    """
    Recommends restaurants based on content-based filtering using the categories column.

    Parameters:
    - df (pd.DataFrame): DataFrame containing restaurant data.
    - restaurant_name (str): Name of the restaurant to base recommendations on.
    - top_n (int): Number of recommendations to return.

    Returns:
    - pd.DataFrame: DataFrame containing recommended restaurants with their similarity scores.
    """
    if 'categories' not in df.columns:
        raise ValueError("DataFrame must contain a 'categories' column.")

    # Fill NaN categories with empty strings
    df['categories'] = df['categories'].fillna('')

    # Ensure there are non-empty categories
    if df['categories'].str.strip().eq('').all():
        raise ValueError("All entries in the 'categories' column are empty. Cannot compute recommendations.")

    # Reset index to ensure alignment with TF-IDF matrix
    df = df.reset_index(drop=True)

    # Check if the restaurant exists in the DataFrame (case-insensitive match)
    matching_restaurants = df[df['name'].str.lower() == restaurant_name.lower()]
    if matching_restaurants.empty:
        raise ValueError(f"Restaurant '{restaurant_name}' not found in the DataFrame.")

    # Get the exact index of the input restaurant
    restaurant_index = matching_restaurants.index[0]

    # Compute TF-IDF matrix for the 'categories' column
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['categories'])

    if tfidf_matrix.shape[0] == 0:
        raise ValueError("TF-IDF matrix is empty. Check the 'categories' column for valid data.")

    # Debugging information
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Restaurant index: {restaurant_index}")

    # Compute cosine similarity between the input restaurant and all others
    cosine_similarities = linear_kernel(tfidf_matrix[restaurant_index:restaurant_index+1], tfidf_matrix).flatten()

    # Get indices of the top_n most similar restaurants (excluding the input restaurant itself)
    similar_indices = cosine_similarities.argsort()[-top_n-1:-1][::-1]

    if len(similar_indices) == 0:
        raise ValueError("No similar restaurants found.")

    # Build a DataFrame with recommendations
    recommendations = df.iloc[similar_indices].copy()
    recommendations['similarity_score'] = cosine_similarities[similar_indices]

    return recommendations[['name', 'categories', 'latitude', 'longitude', 'similarity_score']]

# Streamlit UI
st.title("Restaurant Recommendation System")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### Data Preview")
    st.dataframe(df.head())

    restaurant_name = st.text_input("Enter the name of a restaurant:")
    top_n = st.number_input("Number of recommendations:", min_value=1, max_value=20, value=5)

    if st.button("Recommend"):
        try:
            recommendations = recommend_restaurants(df, restaurant_name, top_n=top_n)

            st.write("### Recommended Restaurants")
            st.dataframe(recommendations[['name', 'categories', 'similarity_score']])

            # Show locations on a map
            m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
            for _, row in recommendations.iterrows():
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=f"{row['name']}\nScore: {row['similarity_score']:.2f}",
                    tooltip=row['name']
                ).add_to(m)

            st.write("### Map of Recommended Restaurants")
            folium_static(m)

        except ValueError as e:
            st.error(f"Error: {e}")