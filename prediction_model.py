import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
from data_utils import get_past_cases, get_law_provisions
from ml_predict import predict_with_ml

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama3-70b-8192")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file.")

client = Groq(api_key=GROQ_API_KEY)

def build_prompt(scenario, ml_result, num_cases=5, num_laws=5):
    """
    Build the prompt for the LLM using:
    - ML prediction
    - Past cases
    - Laws
    - User scenario
    """
    cases = get_past_cases(num_cases)
    laws = get_law_provisions(num_laws)

    case_text = "\n".join(
        [f"- Case: {c['case_title']} | Issues: {c['issues']} | Decision: {c['decision']}"
         for c in cases if c['case_title']]
    )

    law_text = "\n".join(
        [f"- Article {l['article']}: {l['title']} — {l['description']}"
         for l in laws if l['article']]
    )

    prompt = f"""
You are a legal analyst. You have access to:
1. Machine Learning prediction based on past judgments.
2. Relevant past cases and Indian laws.

ML Model Output:
- Predicted Decision: {ml_result['predicted_decision']}
- Confidence: {ml_result['confidence']}%

Past Cases:
{case_text}

Relevant Indian Laws:
{law_text}

Scenario:
{scenario}

TASK:
Write a judgment prediction in STRICT JSON format:
{{
  "summary": "<Brief summary of the scenario>",
  "prediction": "<Final judgment outcome in legal language>",
  "confidence": <integer between 0 and 100>,
  "reasons": ["<reason 1>", "<reason 2>", "<reason 3>"]
}}
"""
    return prompt.strip()

def parse_response(text):
    """Extract JSON from LLM output safely."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {
        "summary": "Unable to parse response.",
        "prediction": text[:200],
        "confidence": 0,
        "reasons": []
    }

def predict_judgment(scenario):
    """Hybrid ML + LLM prediction."""
    # Step 1: ML Prediction
    ml_result = predict_with_ml(scenario)

    # Step 2: Build prompt
    prompt = build_prompt(scenario, ml_result)

    # Step 3: Call Groq API
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800
    )

    raw_output = response.choices[0].message.content
    llm_result = parse_response(raw_output)

    # Step 4: Merge ML confidence into final output if missing
    if not llm_result.get("confidence"):
        llm_result["confidence"] = ml_result["confidence"]

    return llm_result

if __name__ == "__main__":
    scenario_text = "A citizen was detained without trial for 6 months under preventive detention laws."
    result = predict_judgment(scenario_text)
    print(json.dumps(result, indent=2))
