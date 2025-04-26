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

All saved in `/images/` folder.

---

##  Results

- Model successfully predicts ratings with good accuracy.
- Director and genre-based engineered features were highly impactful.
- Random Forest was chosen for its performance and interpretability.


##  How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/Movie-Rating-Prediction-ML.git
   cd Movie-Rating-Prediction-ML
