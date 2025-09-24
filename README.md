# Multimedia-Reecommendation-engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)  
[![Flask](https://img.shields.io/badge/Flask-Backend-green.svg)](https://flask.palletsprojects.com/)  
[![Machine Learning](https://img.shields.io/badge/ML-TF--IDF%20%7C%20Cosine%20Similarity-orange)]()  

A **content-based movie recommendation engine** built as a full **Data Science pipeline**:
- Data Collection (Web Scraping + Manual curation)  
- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature Engineering (TF-IDF Vectorization)  
- Model Building (Cosine Similarity)  
- Deployment (Flask Web App)

---

## 🚀 Features
- ✅ Recommend by **Title similarity**  
- ✅ Recommend by **Genre similarity**  
- ✅ **Hybrid**: Genre + Rating re-ranking  
- ✅ **Random picks** (Surprise Me)  
- ✅ **Popular picks** (Trending actors/genres)  
- 🎥 Displays **Title, Rating, Duration, Genre, Cast, and Poster image**

---

## 🧑‍🔬 Data Science Workflow

### 🔹 1. Data Collection
- Web-scraped movie details (title, genre, rating, duration, cast, poster) using **BeautifulSoup (bs4)**  
- Supplemented missing data with manual curation  

### 🔹 2. Data Cleaning
- Removed duplicates and irrelevant entries  
- Filled missing values with defaults  
- Standardized inconsistent genres (e.g., `"Drama,Action"` → `"Drama, Action"`)  
- Converted all text to lowercase for uniformity  

### 🔹 3. Exploratory Data Analysis (EDA)
- Visualized distribution of genres, ratings, durations  
- Identified most frequent actors and popular genres  
- Checked for outliers and missing data patterns  

### 🔹 4. Feature Engineering
- Built **two separate vector spaces** using **TF-IDF Vectorizer**:  
  - Genre vector space  
  - Movie title vector space  
- Engineered hybrid feature: **Genre similarity + Rating score**  

### 🔹 5. Model Building
- Used **Cosine Similarity** to measure closeness between vectors  
- Modes: Title, Genre, Hybrid, Random, Popular  

### 🔹 6. Deployment
- Built a Flask backend with API endpoints (`/movies`, `/recommend`)  
- Designed an interactive frontend with HTML/CSS/JS  
- Integrated posters and metadata into results view  

---

## 🛠️ Tech Stack
- **Languages/Frameworks**: Python, Flask, HTML, CSS, JS  
- **Libraries**: Pandas, NumPy, Scikit-learn, BeautifulSoup (bs4)  
- **Tools**: Jupyter Notebook (EDA), Excel (dataset), Git/GitHub  
