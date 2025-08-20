import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from data_utils import load_dataset

# Use Path to avoid backslash issues on Windows
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)  # create folder if not exists
MODEL_PATH = MODEL_DIR / "legal_classifier.pkl"

def train_model():
    # Load and clean dataset
    fj_df = load_dataset(
        "fj.csv",
        drop_columns=["judges name(s)", "cited cases", "Unnamed: 0"]
    )

    # Remove rows with missing data
    fj_df = fj_df.dropna(subset=["case title", "issues", "decision"])

    # Combine case title + issues into one text feature
    fj_df["text"] = fj_df["case title"] + " " + fj_df["issues"]

    X = fj_df["text"]
    y = fj_df["decision"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create pipeline (TF-IDF + Logistic Regression)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=200))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {acc:.2%}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
