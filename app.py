# app.py
import streamlit as st
from pathlib import Path
from transformers import pipeline

# -------- Settings --------
CONTEXT_PATH = Path("sample_context.txt")  # fixed source of truth

@st.cache_resource
def get_qa():
    # DistilBERT fine-tuned on SQuAD (no training here; just inference on your file)
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@st.cache_data
def load_context(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()

st.title("Mini QA (Fixed Context from sample_context.txt)")
st.caption("The model answers ONLY from the local file: sample_context.txt")

# Load context from file
context = load_context(CONTEXT_PATH)

# File status
if not context:
    st.error("sample_context.txt not found or is empty. Put your 200+ word Education text in this file and reload.")
    if st.button("Reload file"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

# Context preview & utilities
with st.expander("📄 View context from sample_context.txt"):
    st.write(context)

st.write(f"**Context length:** {len(context)} characters")
col1, col2 = st.columns(2)
with col1:
    if st.button("🔄 Reload file"):
        st.cache_data.clear()
        st.rerun()
with col2:
    st.download_button("⬇️ Download context", data=context, file_name="sample_context.txt")

# Example questions help
with st.expander("📌 Example questions you can ask"):
    st.markdown(
        """
- What is the main focus of modern education?
- What challenge creates the digital divide?
- How are assessments changing in education?
- Which skills are emphasized for students?
- What is the future goal of education?
        """
    )

# Question input (only this is user-editable)
question = st.text_input(
    "❓ Enter your question (the answer will be extracted from sample_context.txt)",
    placeholder="Example: What challenge creates the digital divide?"
)

# Answering
if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        qa = get_qa()
        result = qa(question=question, context=context)
        st.subheader("✅ Answer")
        st.write(result["answer"])
        st.caption(f"Confidence: {result['score']:.3f}")

        # Optional: show the evidence span
        start, end = result.get("start", None), result.get("end", None)
        if isinstance(start, int) and isinstance(end, int):
            with st.expander("🔍 Evidence snippet from context"):
                snippet_pad = 120
                left = max(0, start - snippet_pad)
                right = min(len(context), end + snippet_pad)
                snippet = context[left:right]
                st.write(snippet.replace(context[start:end], f"**{context[start:end]}**"))
