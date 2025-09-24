import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load dataset
movie_df = pd.read_excel("Movies detail with img.xlsx")

# Fill missing values
movie_df["Genre"] = movie_df["Genre"].fillna("")
movie_df["Movie Title"] = movie_df["Movie Title"].fillna("")

# ---------- TF-IDF vectorizers ----------
# Vectorizer for Genre
tfidf_genre = TfidfVectorizer()
vectors_genre = tfidf_genre.fit_transform(movie_df["Genre"])

# Vectorizer for Movie Title
tfidf_title = TfidfVectorizer()
vectors_title = tfidf_title.fit_transform(movie_df["Movie Title"])

# Index mapping for movie title lookup
indexed = pd.Series(data=movie_df.index, index=movie_df['Movie Title'])

# ---------- Helper: Format results ----------
def format_results(indices):
    results = []
    for i in indices:
        raw_img = str(movie_df["img"].iloc[i]) if "img" in movie_df.columns else None
        img_val = None
        if raw_img and raw_img != "nan":
            if raw_img.startswith("http"):
                img_val = raw_img

        results.append({
            "Movie Title": movie_df["Movie Title"].iloc[i],
            "Rating": str(movie_df["Rating"].iloc[i]) if "Rating" in movie_df.columns else "N/A",
            "Duration": str(movie_df["Duration"].iloc[i]) if "Duration" in movie_df.columns else "N/A",
            "Genre": movie_df["Genre"].iloc[i],
            "Cast": movie_df["Cast"].iloc[i] if "Cast" in movie_df.columns else "N/A",
            "Image": img_val
        })
    return results

# ---------- Recommendation function ----------
def mov_rec(name, n, mode="title"):
    """
    mode = "title" → recommend based on movie title similarity
    mode = "genre" → recommend based on genre similarity
    """

    if name not in indexed:
        return {"error": f"Movie '{name}' not found in database."}

    idx = indexed[name]

    if mode == "title":
        # Similarity based on movie title
        dis = linear_kernel(vectors_title[idx], vectors_title).flatten()
        scores = pd.Series(dis).sort_values(ascending=False)
        scores = scores.drop(idx, errors="ignore")  # remove the same movie
        return format_results(scores.index[:n])

    elif mode == "genre":
        # Similarity based on genre of the given movie
        movie_genre = movie_df.loc[idx, "Genre"]
        genre_vec = tfidf_genre.transform([movie_genre])
        dis = linear_kernel(genre_vec, vectors_genre).flatten()
        scores = pd.Series(dis).sort_values(ascending=False)
        scores = scores.drop(idx, errors="ignore")
        return format_results(scores.index[:n])

    else:
        return {"error": "Invalid mode. Use 'title' or 'genre'."}

# ---------- Flask App ----------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("indexchoose.html")

@app.route("/movies")
def movie_list():
    return jsonify(list(movie_df["Movie Title"].dropna().unique()))

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    name = data.get("movie", "").strip()
    mode = data.get("mode", "title")  # either "title" or "genre"

    # Validate number of recommendations
    try:
        n = int(data.get("number", 5))
        n = max(1, min(n, 20))
    except (ValueError, TypeError):
        n = 5

    result = mov_rec(name, n, mode=mode)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
