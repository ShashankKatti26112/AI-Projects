import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset with correct encoding
df = pd.read_csv("C:/Users/hp/Desktop/AI Projects/Ai Project-1/spam.csv", encoding='ISO-8859-1')

# Select relevant columns and rename them
df = df[['v1', 'v2']]
df.columns = ['label', 'message']  # Rename for consistency

# Convert labels to numerical format
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the dataset into features and labels
X = df['message']
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_vec)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")