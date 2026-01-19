# =====================================
# Audible Insights - Streamlit App
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Audible Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎧 Audible Insights: Intelligent Book Recommendations")

# -----------------------------
# Load and Clean Data
# -----------------------------
@st.cache_data
def load_and_clean_data():
    # Paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    catalog_path = os.path.join(base_path, "Audible_Catalog.csv")
    adv_path = os.path.join(base_path, "Audible_Catalog_Advanced_Features.csv")

    # Load datasets
    df_catalog = pd.read_csv(catalog_path)
    df_adv = pd.read_csv(adv_path)

    # Standardize columns names: strip, lower
    df_catalog.columns = df_catalog.columns.str.strip().str.lower()
    df_adv.columns = df_adv.columns.str.strip().str.lower()

    rename_map = {
        "book name": "book_name",
        "number of reviews": "num_reviews",
        "listening time": "listening_time",
        "ranks and genre": "genre"
    }
    df_catalog.rename(columns=rename_map, inplace=True)
    df_adv.rename(columns=rename_map, inplace=True)

    # Merge datasets on book_name and author
    df = pd.merge(
        df_catalog,
        df_adv,
        on=["book_name", "author"],
        how="left",
        suffixes=("_cat", "_adv")
    )

    # Handle overlapping columns:
    # Rating
    df["rating"] = df["rating_cat"].fillna(df["rating_adv"])
    df["rating"] = df["rating"].fillna(df["rating"].median())

    # Number of Reviews
    df["num_reviews"] = df["num_reviews_cat"].fillna(df["num_reviews_adv"])
    df["num_reviews"] = df["num_reviews"].fillna(0).astype(int)

    # Price
    if "price_cat" in df.columns and "price_adv" in df.columns:
        df["price"] = df["price_cat"].fillna(df["price_adv"])
    elif "price_cat" in df.columns:
        df["price"] = df["price_cat"]
    elif "price_adv" in df.columns:
        df["price"] = df["price_adv"]

    df["price"] = df["price"].fillna(df["price"].mean())

    # Description
    if "description" in df.columns:
        df["description"] = df["description"].fillna("No description available")
    else:
        df["description"] = "No description available"

    # Listening Time
    if "listening_time" in df.columns:
        df["listening_time"] = df["listening_time"].fillna("0 hrs")
    else:
        df["listening_time"] = "0 hrs"

    # Book and Author formatting
    df["book_name"] = df["book_name"].str.strip()
    df["author"] = df["author"].str.strip()

    # Remove duplicates
    df.drop_duplicates(subset=["book_name", "author"], inplace=True)

    # Drop extra columns
    df.drop(
        columns=[
            "rating_cat", "rating_adv",
            "num_reviews_cat", "num_reviews_adv",
            "price_cat", "price_adv"
        ],
        inplace=True,
        errors="ignore"
    )

    # -----------------------------
    # Extract Clean Genres
    # -----------------------------
    def extract_genres(text):
        if pd.isna(text) or not isinstance(text, str):
            return "unknown"
        
        parts = [part.strip() for part in text.split(",")]
        
        genres = []
        for part in parts:
            match = re.search(r"in\s+(.+)", part)
            if match:
                genre = match.group(1).strip()
                exclude_phrases = [
                    "audible audiobooks",
                    "originals",
                    "top 100",
                    "audible audiobook",
                    "audiobooks & originals",
                    "audible originals",
                    "books"
                ]
                if not any(excl in genre.lower() for excl in exclude_phrases):
                    genres.append(genre)
        
        if genres:
            unique_genres = sorted(set(genres))
            return ", ".join(unique_genres)
        else:
            return "unknown"

    # Create a clean genre column with parsed genres
    df["genre_clean"] = df["genre"].apply(extract_genres)

    return df

# Load data once
df = load_and_clean_data()

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Search Book", "Genre-Based Recommendation", "Similar Books", "EDA"]
)

# -----------------------------
# Home Page
# -----------------------------
if page == "Home":
    st.subheader("📚 Project Overview")
    st.write("""
    **Audible Insights** is an intelligent book recommendation system that helps users:
    - Discover books based on genres and preferences
    - Find similar books using NLP techniques
    - Explore trends using interactive EDA

    **Tech Stack:**  
    Python | NLP | Machine Learning | Streamlit
    """)

    st.metric("Total Books", df.shape[0])
    st.metric("Total Authors", df["author"].nunique())
    st.metric("Total Genres", df["genre_clean"].nunique())

# -----------------------------
# Search Book Page
# -----------------------------
elif page == "Search Book":
    st.subheader("🔍 Search Book Details")
    book_name = st.selectbox("Select a Book", sorted(df["book_name"].unique()))
    book_data = df[df["book_name"] == book_name].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Author:** {book_data['author']}")
        st.write(f"**Genre:** {book_data['genre_clean']}")
        st.write(f"**Rating:** ⭐ {book_data['rating']}")
        st.write(f"**Price:** ${book_data['price']:.2f}")

    with col2:
        st.write(f"**Reviews:** {book_data['num_reviews']}")
        st.write("**Description:**")
        st.write(book_data["description"])
        st.write(f"**Listening Time:** {book_data['listening_time']}")

# -----------------------------
# Genre-Based Recommendation
# -----------------------------
elif page == "Genre-Based Recommendation":
    st.subheader("🎯 Genre-Based Recommendations")
    genre = st.selectbox("Select Genre", sorted(df["genre_clean"].unique()))
    top_books = (
        df[df["genre_clean"] == genre]
        .sort_values(by=["rating", "num_reviews"], ascending=False)
        .head(5)
    )

    st.write("### 📖 Top 5 Recommended Books")
    st.dataframe(
        top_books[["book_name", "author", "rating", "num_reviews", "price"]],
        use_container_width=True
    )

# -----------------------------
# Similar Books (Content-Based)
# -----------------------------
elif page == "Similar Books":
    st.subheader("🤖 Find Similar Books (NLP Based)")

    # TF-IDF on Description
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["description"])

    selected_book = st.selectbox("Select a Book", df["book_name"].unique())
    book_index = df[df["book_name"] == selected_book].index[0]

    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix)[book_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    book_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[book_indices]

    st.write("### 📚 Similar Books You May Like")
    st.dataframe(
        recommendations[["book_name", "author", "genre_clean", "rating"]],
        use_container_width=True
    )

# -----------------------------
# EDA Page
# -----------------------------
elif page == "EDA":
    st.subheader("📊 Exploratory Data Analysis")

    # 1. Most Popular Genres
    st.write("### 1️⃣ Most Popular Genres")
    genre_count = df["genre_clean"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=genre_count.values, y=genre_count.index, ax=ax, palette="viridis")
    ax.set_xlabel("Number of Books")
    ax.set_ylabel("Genre")
    st.pyplot(fig)

    # 2. Authors with Highest-Rated Books
    st.write("### 2️⃣ Top Authors by Average Rating (min 5 books)")
    author_avg_rating = df.groupby("author").agg(
        avg_rating=("rating", "mean"),
        count_books=("book_name", "count")
    ).query("count_books >= 5").sort_values("avg_rating", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x="avg_rating", y=author_avg_rating.index, data=author_avg_rating, palette="coolwarm", ax=ax)
    ax.set_xlabel("Average Rating")
    ax.set_ylabel("Author")
    st.pyplot(fig)

    # 3. Average Rating Distribution
    st.write("### 3️⃣ Rating Distribution Across Books")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(df["rating"], bins=20, kde=True, color="skyblue", ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Books")
    st.pyplot(fig)

    # 4. Ratings vs Number of Reviews
    st.write("### 4️⃣ Ratings vs Review Counts")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(x="num_reviews", y="rating", data=df, alpha=0.6, ax=ax)
    ax.set_xscale("log")  # Reviews can be skewed
    ax.set_xlabel("Number of Reviews (log scale)")
    ax.set_ylabel("Rating")
    st.pyplot(fig)

    # 5. Publication Year Trends
    if "publication_year" in df.columns:
        st.write("### 5️⃣ Publication Year Trends for Popular Books")
        recent_books = df[df["num_reviews"] >= df["num_reviews"].quantile(0.75)]
        year_count = recent_books.groupby("publication_year").size()
        fig, ax = plt.subplots(figsize=(8,4))
        sns.lineplot(x=year_count.index, y=year_count.values, marker="o", ax=ax)
        ax.set_xlabel("Publication Year")
        ax.set_ylabel("Number of Popular Books")
        st.pyplot(fig)

    # Medium Level Analysis
    st.write("### 6️⃣ Book Clusters Based on Descriptions")
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = tfidf.fit_transform(df["description"])

    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(tfidf_matrix)

    cluster_counts = df["cluster"].value_counts()
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="magma", ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Books")
    st.pyplot(fig)

    st.write("### 7️⃣ How Genre Similarity Affects Recommendations")
    genre_rating = df.groupby("genre_clean").agg(avg_rating=("rating", "mean"), count_books=("book_name","count")).sort_values("avg_rating", ascending=False)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(x="count_books", y="avg_rating", data=genre_rating, hue="avg_rating", palette="viridis", size="count_books", sizes=(50,300), ax=ax)
    ax.set_xlabel("Number of Books in Genre")
    ax.set_ylabel("Average Rating")
    st.pyplot(fig)

    st.write("### 8️⃣ Author Popularity vs Book Ratings")
    author_pop = df.groupby("author").agg(avg_rating=("rating","mean"), total_books=("book_name","count")).sort_values("total_books", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(x="total_books", y="avg_rating", data=author_pop, hue="avg_rating", palette="coolwarm", s=100, ax=ax)
    ax.set_xlabel("Number of Books by Author")
    ax.set_ylabel("Average Rating")
    st.pyplot(fig)

    st.write("### 9️⃣ Ratings by Review Count Distribution")
    df["review_bin"] = pd.qcut(df["num_reviews"].rank(method="first"), 5, labels=["Very Low","Low","Medium","High","Very High"])
    review_rating = df.groupby("review_bin")["rating"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x="review_bin", y="rating", data=review_rating, palette="Set2", ax=ax)
    ax.set_xlabel("Review Count Bin")
    ax.set_ylabel("Average Rating")
    st.pyplot(fig)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("✅ **Audible Insights | Recommendation System Project**")
