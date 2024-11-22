import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# Load your dataset
data = pd.read_csv('output_with_cipher_suite.csv')  # Replace with your dataset path

# Separating features and target variable
X = data.drop('device_name', axis=1)
y = data['device_name']

# Encoding the target variable if it's categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predicting on the test set
y_pred = rf_classifier.predict(X_test)
y_pred_proba = rf_classifier.predict_proba(X_test)  # Prediction probabilities for confidence level

# Calculate the confusion matrix with unique labels in y_test
conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
# Normalize the confusion matrix to get percentages
conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1, keepdims=True) * 100

# Calculate accuracy for train and test sets
train_accuracy = accuracy_score(y_train, rf_classifier.predict(X_train)) * 100
test_accuracy = accuracy_score(y_test, y_pred) * 100

# Calculate RRSE (Root Relative Squared Error)
mse = mean_squared_error(y_test, y_pred)
rrse = (np.sqrt(mse) / np.sqrt(mean_squared_error(y_test, [np.mean(y_test)] * len(y_test)))) * 100

# Create DataFrame for the confusion matrix
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=label_encoder.inverse_transform(np.unique(y_test)),
    columns=label_encoder.inverse_transform(np.unique(y_test))
)

# Create a DataFrame for the confusion matrix with percentages
conf_matrix_percent_df = pd.DataFrame(
    conf_matrix_percent,
    index=label_encoder.inverse_transform(np.unique(y_test)),
    columns=label_encoder.inverse_transform(np.unique(y_test))
)

# Create DataFrame for metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'RRSE'],
    'Value': [test_accuracy, rrse]
})

# Saving the confusion matrix and metrics to separate CSV files
conf_matrix_df.to_csv('rf_confusion_matrix.csv', index=True)
metrics_df.to_csv('rf_results.csv', index=False)

print("Accuracy:", test_accuracy)
print("RRSE:", rrse)
print("Confusion Matrix and metrics saved to 'confusion_matrix.csv' and 'results.csv'")

# Plot 1: Impact of Training (Training vs. Testing Accuracy Plot)
plt.figure(figsize=(8, 6))
plt.bar(['Training Accuracy', 'Testing Accuracy'], [train_accuracy, test_accuracy], color=['blue', 'orange'])
plt.ylabel('Accuracy (%)')
plt.title('Impact of Training: Training vs Testing Accuracy')
plt.savefig('rf_impact_of_training.png')
plt.close()

# Plot 2: Confidence-Level for Correct/Incorrect Classification
# Determine the confidence level for correct and incorrect classifications
correct_classifications = y_test == y_pred
confidence_correct = y_pred_proba[correct_classifications].max(axis=1)
confidence_incorrect = y_pred_proba[~correct_classifications].max(axis=1)

plt.figure(figsize=(10, 6))
sns.histplot(confidence_correct, color='green', label='Correctly Classified', kde=True, bins=10)
sns.histplot(confidence_incorrect, color='red', label='Incorrectly Classified', kde=True, bins=10)
plt.xlabel('Prediction Confidence Level')
plt.ylabel('Frequency')
plt.title('Confidence-Level for Correct/Incorrect Classification')
plt.legend()
plt.savefig('rf_confidence_level.png')
plt.close()

# # Plot 3: Confusion Matrix in Percentages
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix_percent_df, annot=True, fmt=".2f", cmap='Blues')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix (in %)')
# plt.savefig('confusion_matrix_percent.png')
# plt.close()

# Plot 3: Confusion Matrix in Percentages with improved spacing
plt.figure(figsize=(12, 10))  # Increase figure size for better spacing
sns.heatmap(conf_matrix_percent_df, annot=True, fmt=".1f", cmap='Blues', annot_kws={"size": 10}, 
            cbar_kws={'shrink': 1.0})  # Shrink color bar for spacing, control annotation size
plt.xlabel('Predicted Labels', fontsize=14, labelpad=20)  # Label padding for spacing
plt.ylabel('True Labels', fontsize=14, labelpad=20)
plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotate x-ticks for readability
plt.yticks(rotation=0, fontsize=12)  # Keep y-ticks horizontal
plt.title('Confusion Matrix (in %)', fontsize=16, pad=20)
plt.tight_layout()  # Adjust layout to fit everything neatly
plt.savefig('rf_confusion_matrix_percent.png')
plt.close()

# # Extracting feature importances
# feature_importances = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': rf_classifier.feature_importances_
# }).sort_values(by='Importance', ascending=False)
# # Plotting feature importance
# plt.figure(figsize=(10, 6))
# sns.barplot(data=feature_importances, x='Importance', y='Feature', palette='viridis')
# plt.title('Feature Importance', fontsize=16)
# plt.xlabel('Importance Score', fontsize=12)
# plt.ylabel('Features', fontsize=12)
# plt.tight_layout()
# plt.savefig('rf_attr_importance.png')
# plt.close()


# Extracting feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_classifier.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Categorizing features into "More Important," "Medium Important," and "Less Important"
quantiles = feature_importances['Importance'].quantile([0.33, 0.66])  # Divide into thirds
feature_importances['Category'] = feature_importances['Importance'].apply(
    lambda x: 'More Important' if x >= quantiles[0.66] else
              ('Medium Important' if x >= quantiles[0.33] else 'Less Important')
)

# Plotting feature importance as a vertical bar chart
plt.figure(figsize=(12, 8))
ax = sns.barplot(
    data=feature_importances, 
    x='Feature', 
    y='Importance', 
    hue='Category', 
    dodge=False, 
    palette={
        'More Important': 'green',
        'Medium Important': 'blue',
        'Less Important': 'orange'
    }
)
plt.title('Feature Importance (Categorized)', fontsize=16)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Importance Score', fontsize=12)

# Adding numeric values to the bars
for i in ax.patches:
    ax.text(
        i.get_x() + i.get_width() / 2,  # Position the text at the center of the bar
        i.get_height() + 0.005,  # Position slightly above the bar
        f'{i.get_height():.3f}',  # Format the value with 3 decimal places
        ha='center', va='bottom', fontsize=10
    )

plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.legend(title='Category', loc='upper right')
plt.tight_layout()

# Saving the plot as an image
plt.savefig('rf_ft_importance.png', dpi=300)
plt.close()

print("Plots saved as 'impact_of_training.png', 'confidence_level.png', and 'confusion_matrix.png'")


cv_scores = cross_val_score(rf_classifier, X, y, cv=5)  # 5-fold cross-validation
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores)}")