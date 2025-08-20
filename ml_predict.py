import joblib
from pathlib import Path

MODEL_PATH = Path("models/legal_classifier.pkl")

def predict_with_ml(scenario_text):
    """
    Load the trained model and predict decision + confidence.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Train it first.")

    # Load model
    pipeline = joblib.load(MODEL_PATH)

    # Get prediction and probabilities
    pred = pipeline.predict([scenario_text])[0]
    probs = pipeline.predict_proba([scenario_text])[0]
    confidence = round(max(probs) * 100, 2)  # % confidence

    return {
        "predicted_decision": pred,
        "confidence": confidence
    }

if __name__ == "__main__":
    # Example usage
    test_scenario = "A citizen was detained without trial for 6 months under preventive detention laws."
    result = predict_with_ml(test_scenario)
    print(f"Prediction: {result['predicted_decision']} (Confidence: {result['confidence']}%)")
