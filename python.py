# Importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Setting seaborn style for better aesthetics in plots
sns.set(style="whitegrid")

# Load the dataset
data = pd.read_csv('Movies.csv')

# Displaying the first 10 rows of the dataset
print("Top 10 rows:")
print(data.head(10))

# Displaying the last 10 rows of the dataset
print("\nBottom 10 rows:")
print(data.tail(10))

# Output the shape of the dataset (number of rows and columns)
print("\nNumber of Rows:", data.shape[0])
print("Number of Columns:", data.shape[1])

# Get detailed information about the dataset (data types, non-null counts, etc.)
print("\nDataset Info:")
print(data.info())

# Drop duplicate rows and rows with missing values
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

# Display basic summary statistics of the dataset
print("\nSummary Statistics:")
print(data.describe(include='all'))

# Data cleaning: Clean the 'Votes' column by removing commas and converting it to a numeric type
data['Votes'] = data['Votes'].replace(',', '', regex=True)
data['Votes'] = pd.to_numeric(data['Votes'], errors='coerce')
data.dropna(subset=['Votes'], inplace=True)  # Drop rows where 'Votes' is NaN

# Apply log transformation to the 'Votes' column to reduce skewness
data['Votes_log'] = np.log1p(data['Votes'])

# Feature engineering: Calculate the success rate for each director based on average movie ratings
if 'Director' in data.columns and 'Rating' in data.columns:
    data['Director_Success'] = data.groupby('Director')['Rating'].transform('mean')

# Feature engineering: Calculate the average rating for movies within the same genre
if 'Genre' in data.columns and 'Rating' in data.columns:
    data['Genre_Avg_Rating'] = data.groupby('Genre')['Rating'].transform('mean')

# Check if the 'Actor' column exists and compute average rating for movies by the same actor
if 'Actor' in data.columns and 'Rating' in data.columns:
    data['Actor_Avg_Rating'] = data.groupby('Actor')['Rating'].transform('mean')

# Display the newly engineered features, handling cases where some columns might be missing
try:
    print(data[['Director', 'Genre', 'Rating', 'Director_Success', 'Genre_Avg_Rating', 'Actor_Avg_Rating']].head())
except KeyError as e:
    print(f"Column missing: {e}")
    print(data[['Director', 'Genre', 'Rating', 'Director_Success', 'Genre_Avg_Rating']].head())  # Exclude missing columns

# Encode categorical columns ('Director', 'Genre', 'Actor', 'Language', 'Country') into numerical values
categorical_cols = ['Director', 'Genre', 'Actor', 'Language', 'Country']
for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

# Select numeric features for the model (excluding the 'Rating' column)
features = data.select_dtypes(include=[np.number]).columns.tolist()
features = [col for col in features if col != 'Rating']

# Define the input features (X) and the target variable (y)
X = data[features]
y = data['Rating']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print the model evaluation metrics: RMSE (Root Mean Squared Error) and R² (R-squared)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot 1: Correlation heatmap showing the relationships between features
plt.figure(figsize=(12, 8))
sns.heatmap(data[features + ['Rating']].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Plot 2: Distribution of Movie Ratings as a histogram
plt.figure(figsize=(8, 5))
sns.histplot(data['Rating'], bins=20, kde=True, color="skyblue")
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

# Plot 3: Feature Importance plot from the Random Forest model
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = np.array(features)[indices]

# Custom colors for the bars in the plot
colors = sns.color_palette("husl", len(sorted_features))  # Color palette for individual bars

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=sorted_features, hue=sorted_features, palette=colors, legend=False) 
plt.title("Feature Importances from Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Plot 4: Boxplot showing the distribution of Movie Ratings
plt.figure(figsize=(8, 6))
sns.boxplot(x=data['Rating'], color="skyblue")
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.tight_layout()
plt.show()
