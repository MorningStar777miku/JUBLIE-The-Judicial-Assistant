# Jublie — Full project bundle (multiple modules concatenated)
# -----------------------------------------------------------------
# This single Python file contains the core modules for the Jublie 2.0
# prototype. For convenience I've concatenated multiple files here with
# clear separators. When you copy this into your project, split each
# section into its own file as named in the headers.
# -----------------------------------------------------------------

# ==========================
# File: jublie/pipeline/preprocess_and_retrieval.py
# ==========================
"""
Preprocessing & Dual Retrieval
- maps fj.csv and coi.csv columns explicitly
- builds TF-IDF, embeddings (sentence-transformers), FAISS indexes
- exposes `build_all()` and `retrieve()`
"""
from pathlib import Path
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import faiss
import joblib

BASE_DIR = Path("jublie")
DATA_DIR = BASE_DIR / "dataset"
INDICES_DIR = BASE_DIR / "indices"
INDICES_DIR.mkdir(parents=True, exist_ok=True)
FJ_CSV = DATA_DIR / "fj.csv"
COI_CSV = DATA_DIR / "coi.csv"
EMBED_MODEL_NAME = "all-mpnet-base-v2"
TOP_K = 5

FJ_INDEX_PATH = INDICES_DIR / "fj.faiss"
COI_INDEX_PATH = INDICES_DIR / "coi.faiss"
FJ_META_PATH = INDICES_DIR / "fj_meta.parquet"
COI_META_PATH = INDICES_DIR / "coi_meta.parquet"
TFIDF_PATH = INDICES_DIR / "tfidf.joblib"


def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\bArt\.\s*(\d+)\b", r"Article \1", s, flags=re.I)
    s = re.sub(r"\bSec\.\s*(\d+)\b", r"Section \1", s, flags=re.I)
    s = s.replace('\u200b', '')
    return s


def load_and_preprocess_fj(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Build _text using specific columns you confirmed
    def build_case_text(row):
        parts = []
        for c in ["case title", "issues", "decision", "cited cases"]:
            if c in row and pd.notna(row[c]):
                parts.append(str(row[c]))
        return clean_text(" ".join(parts))
    df["_text"] = df.apply(build_case_text, axis=1)
    if "case_id" not in df.columns:
        df["case_id"] = df.index.astype(str)
    # ensure decision column exists
    if "decision" not in df.columns:
        raise ValueError("fj.csv must contain 'decision' column for supervised training")
    return df


def load_and_preprocess_coi(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    def build_article_text(row):
        parts = []
        if "article" in row and pd.notna(row["article"]):
            parts.append(f"Article {row['article']}")
        for c in ["title", "description"]:
            if c in row and pd.notna(row[c]):
                parts.append(str(row[c]))
        return clean_text(" ".join(parts))
    df["_text"] = df.apply(build_article_text, axis=1)
    if "law_id" not in df.columns:
        df["law_id"] = df.index.astype(str)
    return df


def embed_texts(texts: list, model_name: str = EMBED_MODEL_NAME, model_obj=None):
    if model_obj is None:
        model_obj = SentenceTransformer(model_name)
    emb = model_obj.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    emb = normalize(emb)
    return emb, model_obj


def build_faiss_index(embeddings: np.ndarray):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    return index


def save_index(index, path: Path):
    faiss.write_index(index, str(path))


def load_index(path: Path):
    if not path.exists():
        return None
    return faiss.read_index(str(path))


def build_all(fj_path=FJ_CSV, coi_path=COI_CSV, embed_model_name=EMBED_MODEL_NAME):
    print("Loading and preprocessing CSVs...")
    fj = load_and_preprocess_fj(fj_path)
    coi = load_and_preprocess_coi(coi_path)
    print(f"FJ rows: {len(fj)}, COI rows: {len(coi)}")
    print("Fitting TF-IDF vectorizer...")
    corpus = list(fj["_text"].astype(str).tolist()) + list(coi["_text"].astype(str).tolist())
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    tfidf.fit(corpus)
    joblib.dump(tfidf, TFIDF_PATH)
    print("Building embeddings (may take time)...")
    fj_emb, model_obj = embed_texts(fj["_text"].tolist(), model_name=embed_model_name)
    coi_emb, _ = embed_texts(coi["_text"].tolist(), model_obj=model_obj)
    print("Building FAISS indexes...")
    fj_index = build_faiss_index(fj_emb)
    coi_index = build_faiss_index(coi_emb)
    print("Saving indexes and metadata...")
    save_index(fj_index, INDICES_DIR / "fj.faiss")
    save_index(coi_index, INDICES_DIR / "coi.faiss")
    fj.to_parquet(INDICES_DIR / "fj_meta.parquet", index=False)
    coi.to_parquet(INDICES_DIR / "coi_meta.parquet", index=False)
    (INDICES_DIR / "embed_model.name.txt").write_text(embed_model_name)
    print("Build complete.")


def retrieve(query: str, top_k_cases: int = TOP_K, top_k_laws: int = TOP_K):
    if not (INDICES_DIR / "fj.faiss").exists():
        raise FileNotFoundError("Indexes not found. Run build_all() first")
    tfidf = joblib.load(TFIDF_PATH)
    fj_meta = pd.read_parquet(INDICES_DIR / "fj_meta.parquet")
    coi_meta = pd.read_parquet(INDICES_DIR / "coi_meta.parquet")
    fj_index = load_index(INDICES_DIR / "fj.faiss")
    coi_index = load_index(INDICES_DIR / "coi.faiss")
    model_name = (INDICES_DIR / "embed_model.name.txt").read_text().strip()
    model_obj = SentenceTransformer(model_name)
    q_toks = tfidf.transform([query])
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = q_toks.toarray().ravel()
    topn_ids = coefs.argsort()[-10:][::-1]
    top_keywords = feature_names[topn_ids][coefs[topn_ids] > 0].tolist()
    q_emb = model_obj.encode([clean_text(query)], convert_to_numpy=True)
    q_emb = normalize(q_emb)
    q_emb = q_emb.astype(np.float32)
    D_c, I_c = fj_index.search(q_emb, top_k_cases)
    cases = []
    for score, idx in zip(D_c[0], I_c[0]):
        meta = fj_meta.iloc[int(idx)].to_dict()
        cases.append({"score": float(score), "index": int(idx), "meta": meta})
    D_l, I_l = coi_index.search(q_emb, top_k_laws)
    laws = []
    for score, idx in zip(D_l[0], I_l[0]):
        meta = coi_meta.iloc[int(idx)].to_dict()
        laws.append({"score": float(score), "index": int(idx), "meta": meta})
    return {"cases": cases, "laws": laws, "tfidf_keywords": top_keywords}


# ==========================
# File: jublie/pipeline/train_baseline.py
# ==========================
"""
Train a baseline classifier using TF-IDF + Logistic Regression.
Saves: models/baseline_model.joblib and models/tfidf_vectorizer.joblib
"""
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_baseline(fj_csv_path=FJ_CSV):
    df = pd.read_csv(fj_csv_path)
    # Build input text
    texts = []
    for _, row in df.iterrows():
        parts = []
        for c in ["case title", "issues", "cited cases"]:
            if c in row and pd.notna(row[c]):
                parts.append(str(row[c]))
        texts.append(clean_text(" ".join(parts)))
    y = df["decision"].astype(str).str.strip()
    # label encode
    labels = {v: i for i, v in enumerate(sorted(y.unique()))}
    y_enc = y.map(labels).astype(int)
    # tfidf
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    X = tfidf.fit_transform(texts)
    X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    print("Training baseline LogisticRegression...")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Validation accuracy: {acc:.4f}")
    print(classification_report(y_val, preds))
    joblib.dump(clf, MODELS_DIR / "baseline_model.joblib")
    joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.joblib")
    # save label map
    (MODELS_DIR / "label_map.json").write_text(json.dumps(labels))
    print("Saved baseline model and tfidf vectorizer.")


# ==========================
# File: jublie/pipeline/llm_formatter.py
# ==========================
"""
Groq integration + prompt templating
Requires 'groq' package and GROQ_API_KEY in environment or config
"""
import os
import json
try:
    import groq
except Exception:
    groq = None

GROQ_MODEL = "mixtral-8x7b-32768"


def build_groq_prompt(query: str, pred_label: str, confidence: float, cases: list, laws: list):
    # Create compact representations of cases and laws
    short_cases = []
    for c in cases:
        m = c['meta']
        short_cases.append(f"{m.get('case title','')} — issues: {m.get('issues','')}; decision: {m.get('decision','')}")
    short_laws = []
    for l in laws:
        m = l['meta']
        short_laws.append(f"{m.get('article','')} {m.get('title','')}: { (m.get('description','') or '')[:200] }")
    prompt = f"""
You are a legal analyst assistant. Produce a structured prediction output.
Context:
Relevant Laws:
{chr(10).join(['- '+s for s in short_laws])}

Past Similar Cases:
{chr(10).join(['- '+s for s in short_cases])}

Scenario:
{query}

Predicted Outcome: {pred_label}
Confidence: {confidence:.2f}

Please format the result EXACTLY as:
📜 Case Summary:\n<one-paragraph summary>\n\n🔮 Prediction:\n<single-sentence prediction>\n\n📊 Confidence:\n<percentage>\n\n📌 Reasons:\n- reason 1\n- reason 2\n- reason 3\n
Keep language formal, concise, and grounded in the provided context.
"""
    return prompt


def call_groq(prompt: str, api_key: str = None):
    if groq is None:
        raise RuntimeError("groq package not installed. Install 'groq' to use Groq API integration.")
    if api_key is None:
        api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise RuntimeError("Groq API key not provided. Set GROQ_API_KEY in environment or pass api_key param.")
    client = groq.Client(api_key=api_key)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=800
    )
    content = resp.choices[0].message['content']
    return content


# ==========================
# File: jublie/pipeline/predictor.py
# ==========================
"""
Loads baseline model + tfidf + retrieval. Returns structured JSON.
If GROQ_API_KEY is available, it will call Groq for final humanized text.
"""
import json
import joblib
import os

MODELS_DIR = BASE_DIR / "models"

def load_artifacts():
    clf = joblib.load(MODELS_DIR / "baseline_model.joblib")
    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    label_map = json.loads((MODELS_DIR / "label_map.json").read_text())
    inv_label_map = {v: k for k, v in label_map.items()}
    return clf, tfidf, inv_label_map


def predict_case(query: str, top_k_cases: int = 5, top_k_laws: int = 5, call_llm: bool = True):
    clf, tfidf, inv_label_map = load_artifacts()
    # vectorize
    Xq = tfidf.transform([query])
    probs = clf.predict_proba(Xq)[0]
    pred_idx = int(probs.argmax())
    pred_label = inv_label_map[pred_idx]
    confidence = float(probs.max())
    # retrieval from indices
    from jublie.pipeline.preprocess_and_retrieval import retrieve
    retrieval = retrieve(query, top_k_cases=top_k_cases, top_k_laws=top_k_laws)
    cases = retrieval['cases']
    laws = retrieval['laws']
    result = {
        'case_summary': query,
        'prediction': pred_label,
        'confidence': confidence,
        'reasons': [
            f"Top similar case: {cases[0]['meta'].get('case title','')} (score={cases[0]['score']:.3f})" if cases else "",
            f"Most relevant law: {laws[0]['meta'].get('article','')} {laws[0]['meta'].get('title','')} (score={laws[0]['score']:.3f})" if laws else ""
        ],
        'retrieval': retrieval
    }
    # Call Groq to produce humanized output if requested
    if call_llm:
        try:
            from jublie.pipeline.llm_formatter import build_groq_prompt, call_groq
            prompt = build_groq_prompt(query, pred_label, confidence, cases, laws)
            groq_api_key = os.getenv('GROQ_API_KEY')
            human_text = call_groq(prompt, api_key=groq_api_key)
            result['human_readable'] = human_text
        except Exception as e:
            result['human_readable_error'] = str(e)
    return result


# ==========================
# File: jublie/api/app.py
# ==========================
"""
FastAPI server exposing /predict endpoint
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Jublie API")

class PredictRequest(BaseModel):
    text: str
    top_k: int = 5
    call_llm: bool = True

@app.post('/predict')
async def predict(req: PredictRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="text is required")
    try:
        from jublie.pipeline.predictor import predict_case
        out = predict_case(req.text, top_k_cases=req.top_k, top_k_laws=req.top_k, call_llm=req.call_llm)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)


# ==========================
# File: jublie/config.py
# ==========================
GROQ_API_KEY = ""  # or set via environment variable GROQ_API_KEY


# ==========================
# File: requirements.txt (content below, save as text file)
# ==========================
requirements_txt = '''
pandas
numpy
scikit-learn
sentence-transformers
faiss-cpu
joblib
fastapi
uvicorn
groq
python-multipart
pydantic
'''

# ==========================
# File: README.md (short)
# ==========================
readme = '''
Jublie 2.0 — Dual-source legal prediction prototype

Setup:
1. Create virtual env: python -m venv .venv && source .venv/bin/activate
2. pip install -r requirements.txt
3. Place fj.csv and coi.csv in jublie/dataset/
4. Build indexes: python -c "from jublie.pipeline.preprocess_and_retrieval import build_all; build_all()"
5. Train baseline: python -c "from jublie.pipeline.train_baseline import train_baseline; train_baseline()"
6. Run API: python jublie/api/app.py

Set GROQ_API_KEY as env var to enable Groq outputs.
'''

# Save helper to split file automatically if desired
if __name__ == '__main__':
    # write the concatenated package as a convenience file, but also save the auxiliary files
    root = Path.cwd() / 'out_bundle'
    root.mkdir(exist_ok=True)
    (root / 'requirements.txt').write_text(requirements_txt)
    (root / 'README.md').write_text(readme)
    print(f"Wrote project bundle helpers to {root}")
    print("Split the bundle into separate modules as needed.")
