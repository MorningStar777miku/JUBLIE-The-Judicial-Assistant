import streamlit as st
from module.login import login_ui
from module.law_chatbot import show_law_chatbot
from module.general_chatbot import show_general_chatbot
from prediction_model import predict_judgment
import docx
import PyPDF2

# ---------------------
# Helper Functions
# ---------------------
def extract_text_from_file(uploaded_file):
    """Extracts text from uploaded txt, docx, or pdf."""
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in reader.pages])
    else:
        return ""

def show_prediction_ui():
    """UI for hybrid ML + LLM prediction inside Law-jublie."""
    st.subheader("⚖️ Legal Judgment Prediction")
    scenario_text = st.text_area("📝 Enter scenario description", height=200)
    uploaded_file = st.file_uploader("📂 Or upload a case file (TXT, DOCX, PDF)", type=["txt", "docx", "pdf"])

    if uploaded_file:
        file_text = extract_text_from_file(uploaded_file)
        scenario_text = (scenario_text + "\n" + file_text) if scenario_text else file_text

    if st.button("🔮 Predict Judgment"):
        if not scenario_text.strip():
            st.error("Please enter a scenario or upload a file.")
        else:
            with st.spinner("Analyzing scenario..."):
                result = predict_judgment(scenario_text)

            st.subheader("📜 Case Summary")
            st.write(result.get("summary", ""))

            st.subheader("🔮 Predicted Outcome")
            st.write(result.get("prediction", ""))

            st.subheader("📊 Confidence")
            st.write(f"{result.get('confidence', 0)}%")

            st.subheader("📌 Reasons")
            for r in result.get("reasons", []):
                st.markdown(f"- {r}")

# ---------------------
# App Configuration
# ---------------------
st.set_page_config(page_title="Jubile Chatbot", layout="wide")

# ---------------------
# Session State Init
# ---------------------
for key, default in {
    "authenticated": False,
    "current_page": "law",
    "username": "",
    "law_messages": [],
    "general_messages": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------
# Authentication
# ---------------------
if not st.session_state.authenticated:
    login_ui()
    st.stop()

# ---------------------
# Navigation Buttons
# ---------------------
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("Law-jublie"):
        st.session_state.current_page = "law"
        st.rerun()
with col2:
    if st.button("Com-jublie"):
        st.session_state.current_page = "general"
        st.rerun()
with col3:
    if st.button("Logout"):
        keys_to_clear = ["authenticated", "username", "law_messages", "general_messages", "current_page"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ---------------------
# Page Rendering
# ---------------------
if st.session_state.current_page == "law":
    show_law_chatbot()           
    st.markdown("---")           # Separator
    show_prediction_ui()         # Integrated ML+LLM prediction
elif st.session_state.current_page == "general":
    show_general_chatbot()
