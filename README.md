Markdown
# 🎬 GMM Soft-Clustering Movie Recommender

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)

A highly nuanced, interactive movie recommendation engine built using **Gaussian Mixture Models (GMM)** and deployed via **Streamlit**. 

Unlike standard recommendation systems that rely on rigid genre matching or simple distance metrics, this engine models the underlying, overlapping distributions of movie features to provide intelligent, blended recommendations.

## 🌟 Features

* **Soft-Clustering Recommendations:** Uses probabilistic cluster assignments rather than rigid categories.
* **Smart Tie-Breaking:** Sorts similarity matches by a combination of Match Score, Popularity, and Vote Average.
* **Blazing Fast UI:** Pre-computed probability matrices and serialized models allow for millisecond load times in the Streamlit web app.
* **Visual Interface:** Automatically fetches and displays movie posters and metadata.

## 🧠 Architectural Choices: Why GMM?

When building the core engine for this recommender, traditional clustering (like K-Means) or standard nearest-neighbor algorithms (like KNN) were intentionally avoided. Here is why:

### The Problem with Hard Clustering (K-Means)
Algorithms like K-Means utilize **Hard Clustering**, meaning every data point (movie) is forced entirely into one specific cluster. 
* *The Flaw:* Movies rarely fit into a single box. A film like *The Batman* is a blend of Action, Crime, Drama, and Mystery. Forcing it strictly into an "Action" cluster means the system loses all nuance of its other genres. 

### The Problem with Direct Nearest Neighbors (KNN)
Standard KNN calculates the direct geometric distance (like Euclidean) between movies in the feature space. 
* *The Flaw:* In a high-dimensional dataset with sparse, one-hot encoded genres, distance metrics often break down (the "Curse of Dimensionality"). It fails to recognize broader, underlying patterns or latent groupings of how genres historically blend together.

### The Solution: Gaussian Mixture Models (Soft Clustering)
GMM solves these issues by treating clusters as overlapping probability distributions.
1. **Overlapping Boundaries:** GMM utilizes **Soft Clustering**. Instead of saying a movie *is* in Cluster 4, it outputs a probability matrix (e.g., this movie is 70% Cluster 4, 20% Cluster 1, and 10% Cluster 7).
2. **Capturing Nuance:** By calculating the **Cosine Similarity** between these precise probability vectors, the engine perfectly captures films that blend genres in the exact same way. If you search for an Action-Comedy, the system finds other movies with the exact same Action-Comedy probability distribution.
3. **Data-Driven Optimization:** The optimal number of clusters (components) was rigorously determined using **BIC (Bayesian Information Criterion)** and **AIC (Akaike Information Criterion)** curves, ensuring the model fits the data organically without overfitting.

## 🛠️ Tech Stack

* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (GMM, StandardScaler, Cosine Similarity)
* **Web Deployment:** `streamlit`
* **Model Serialization:** `joblib`

## 🚀 Running the Project Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/gmm-movie-recommender.git](https://github.com/yourusername/gmm-movie-recommender.git)
cd gmm-movie-recommender
2. Install Dependencies
Ensure you have Python 3.8+ installed, then run:

Bash
pip install -r requirements.txt
3. Ensure Assets are Present
Make sure the following serialized files (generated from the Kaggle notebook) are in the root directory:

gmm_model.pkl

cluster_probabilities.pkl

movies_metadata.pkl

4. Launch the App
Bash
streamlit run app.py
Navigate to http://localhost:8501 in your browser to interact with the engine.