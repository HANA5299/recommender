📂 Datasets Used:

movies.csv

links.csv

ratings.csv

tags.csv

🔎 Step 1: Data Exploration & Cleaning

Checked and handled null values.

Removed duplicate rows.

Performed statistical exploration of ratings and movie features.

Applied TF-IDF vectorization to taglines and metadata for later similarity calculations (cosine similarity).

🤖 Step 2: Model Training

Implemented a recommend_movies() function to calculate similarity scores and rank movies accordingly.

Split the dataset into train and test sets.

Applied SVD (Singular Value Decomposition) for collaborative filtering.

📊 Evaluation Metrics (Collaborative Filtering):

RMSE: 0.8819

MAE: 0.6778

🔗 Step 3: Hybrid Filtering

Combined collaborative filtering (SVD) scores with content-based similarity scores.

Evaluated hybrid model using precision and recall across different α (weighting) values.

Generated hybrid recommendations for User 1.
 
