import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score

# Load the CSV file (replace with your actual file)
data = pd.read_csv('all_attr_output.csv')

# Extract 'dst_port' column and convert it to string
dst_port = data['dst_port'].astype(str)

# Apply TF-IDF on 'dst_port' with n-grams and min_df
vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
dst_port_tfidf = vectorizer.fit_transform(dst_port)

# Define the target variable (label)
y = data['device_name']  # Use 'device_name' as target column

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(dst_port_tfidf, y, test_size=0.2, random_state=42)

# Initialize Naive Bayes classifier with hyperparameter tuning
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Use the best model from grid search
nb = grid_search.best_estimator_

# Cross-validate the model to evaluate accuracy
cross_val_scores = cross_val_score(nb, dst_port_tfidf, y, cv=5)
print("Cross-Validation Accuracy Scores:", cross_val_scores)
print("Mean Cross-Validation Accuracy:", cross_val_scores.mean())

# Make predictions and get confidence values for the entire dataset
predictions = nb.predict(dst_port_tfidf)
confidences = nb.predict_proba(dst_port_tfidf).max(axis=1)  # Max confidence for each sample

# Add predictions and confidence values to the original DataFrame
data['dst_port_class'] = predictions
data['confidence'] = confidences

# Save the updated DataFrame with predictions and confidence values to a new CSV file
output_file = 'output_with_predictions_and_confidence.csv'
data.to_csv(output_file, index=False)

print(f"Output saved to {output_file}")
