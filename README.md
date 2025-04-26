# Movie Rating Prediction using Machine Learning

## Objective
To build a predictive model that estimates movie ratings using attributes such as Director, Genre, Actor, Language, and more. The aim is to use feature engineering and machine learning techniques to understand what impacts movie ratings the most and forecast them accurately.

## Dataset used
-<a href="https://github.com/a-aanchal/Movie-Rating-Prediction/blob/main/Movies.csv">Dataset</a>

## Tech Stack
- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn

##  Approach

1. **Data Exploration & Cleaning**
   - Removed duplicates and missing values.
   - Converted 'Votes' column to numeric after removing commas.

2. **Feature Engineering**
   - Created new features:
     - 'Director_Success' → Average rating per director.
     - 'Genre_Avg_Rating' → Average rating per genre.
     - 'Actor_Avg_Rating' → Average rating per actor.
   - Applied log transformation on 'Votes' to reduce skewness.

3. **Encoding**
   - Used Label Encoding on categorical variables: Director, Genre, Actor, Language, Country.

4. **Modeling**
   - Split the data (80% train / 20% test).
   - Trained a Random Forest Regressor model.

5. **Evaluation**
   - RMSE (Root Mean Squared Error): `XX.XX`
   - R² Score: `X.XX`
   - Visualized feature importance, rating distribution, and correlation heatmap.

##  Visualizations

- Correlation Heatmap
- Histogram of Ratings
- Feature Importance Barplot
- Rating Distribution Boxplot

## images
![Screenshot 2025-04-26 120034](https://github.com/user-attachments/assets/c3c289e4-69d0-4b8f-8d93-305c63ef0343)
![Screenshot 2025-04-26 120059](https://github.com/user-attachments/assets/8c26c3f5-fced-43d3-929c-80e0c8101ec0)
![Screenshot 2025-04-26 120139](https://github.com/user-attachments/assets/a4b7573f-ce41-4505-a835-145c2a65287e)
![Screenshot 2025-04-26 120117](https://github.com/user-attachments/assets/eb1ef815-0ab5-4915-99ea-f68e3de0e8a5)

## Project Insights

-The dataset was thoroughly cleaned, handling duplicates and missing values to ensure data quality.

-Categorical features such as Director, Genre, Actor, Language, and Country were encoded for model compatibility.

-Feature engineering significantly improved prediction accuracy by introducing metrics like Director Success Rate, Genre Average Rating, and Actor Average Rating.

-A Random Forest Regressor was chosen for its robustness and ability to handle mixed data types.

-Achieved strong model performance with R² Score and RMSE used as key evaluation metrics.

-Visualization tools like heatmaps and bar plots provided interpretability and insights into feature importance and data distribution.

##  Results

- Model successfully predicts ratings with good accuracy.
- Director and genre-based engineered features were highly impactful.
- Random Forest was chosen for its performance and interpretability.


##  How to Run

-<a href="https://github.com/a-aanchal/Movie-Rating-Prediction/blob/main/python.py">Codes</a>

