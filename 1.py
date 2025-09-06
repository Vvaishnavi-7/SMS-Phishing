import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (update filename to your Excel file)
df = pd.read_excel("Dataset.xlsx")

# ✅ Rename columns for convenience
df = df.rename(columns={"Message": "message", "Lable": "label"})

# Drop missing values
df = df.dropna(subset=['message', 'label'])

# Features (X) and target (y)
X = df['message']
y = df['label']

# Split data into train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build pipeline: TF-IDF + Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example prediction
# example = ["hfuhjfdncUse data without worrying because now you can get 1GB data loan at no extra cost. Data validity- 1 day. Dial *567*3# or click i.airtel.in/get_data_loan to avail. Get 1GB less with your next data top-up."]
# print("Prediction for example:", model.predict(example))
