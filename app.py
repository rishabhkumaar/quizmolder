"""
app.py - Streamlit UI for QuizMolder

Highlights:
- Choose quiz type and distractor strategy
- Live progress & caching via st.session_state
- Export as JSON / TXT / PDF (PDF optional; uses fpdf if installed)
- Copy to clipboard for quick sharing
"""

import streamlit as st
import json
from utils.quizgen import QuizGenerator
import tempfile
import os
from datetime import datetime

# Optional PDF export
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

st.set_page_config(page_title="QuizMolder Pro", page_icon="🧠", layout="wide")
st.title("🧠 QuizMolder — Hackathon Edition")
st.write("Generate high-quality quizzes with configurable distractors and export options.")

# Sidebar controls
st.sidebar.header("Settings")
quiz_type = st.sidebar.selectbox("Quiz Type", ["Multiple Choice (MCQ)", "True/False", "Short Answer"])
num_questions = st.sidebar.slider("Number of Questions", 1, 30, 6)
num_options = None
distractor_strategy = st.sidebar.selectbox("Distractor Strategy", ["random", "wordnet", "embedding"])
if quiz_type == "Multiple Choice (MCQ)":
    num_options = st.sidebar.slider("Options per Question", 2, 6, 4)

# Example input
if "last_quiz" not in st.session_state:
    st.session_state["last_quiz"] = None

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📥 Paste Notes")
    sample = st.expander("Show sample text")
    sample.write(
        "Python was created by Guido van Rossum and first released in 1991. "
        "The mitochondrion is often called the powerhouse of the cell. "
        "Water freezes at 0°C and boils at 100°C (at 1 atm)."
    )
    text_input = st.text_area("Lecture notes / text", height=300, placeholder="Paste your notes here...")
    generate_btn = st.button("🚀 Generate")

with col2:
    st.subheader("📑 Output")

    if generate_btn:
        if not text_input or not text_input.strip():
            st.warning("Paste some notes first.")
        else:
            # main generation flow
            with st.spinner("Generating quiz..."):
                qg = QuizGenerator(text_input)
                try:
                    if quiz_type == "Multiple Choice (MCQ)":
                        quiz = qg.generate_mcq(num_questions=num_questions, num_options=num_options, distractor_strategy=distractor_strategy)
                    elif quiz_type == "True/False":
                        quiz = qg.generate_true_false(num_questions=num_questions)
                    else:
                        quiz = qg.generate_short_answer(num_questions=num_questions)

                    st.session_state["last_quiz"] = quiz

                except Exception as e:
                    st.error(f"Error while generating: {e}")
                    quiz = []

            if not quiz:
                st.warning("No questions generated — try longer text or different settings.")
            else:
                st.success(f"Generated {len(quiz)} questions.")
                for i, q in enumerate(quiz, 1):
                    with st.expander(f"Q{i}. {q['question']}", expanded=False):
                        if q.get("options"):
                            for opt in q["options"]:
                                st.markdown(f"- {opt}")
                        st.write(f"**Answer:** {q['answer']}")
                        if st.button(f"Copy Answer {i}", key=f"copy_{i}"):
                            st.clipboard_set(q['answer'])
                            st.toast("Answer copied to clipboard")

                # Export options
                st.markdown("---")
                st.subheader("Export / Save")
                js = json.dumps(quiz, indent=2, ensure_ascii=False)
                st.download_button("Download JSON", data=js, file_name=f"quiz_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
                # TXT export
                text_lines = []
                for idx, q in enumerate(quiz, 1):
                    block = f"Q{idx}. {q['question']}\n"
                    if q.get("options"):
                        block += "Options: " + ", ".join(q["options"]) + "\n"
                    block += f"Answer: {q['answer']}\n"
                    text_lines.append(block)
                txt = "\n".join(text_lines)
                st.download_button("Download TXT", data=txt, file_name=f"quiz_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt", mime="text/plain")

                # PDF export (if available)
                if PDF_AVAILABLE:
                    pdf_file = None
                    try:
                        pdf = FPDF()
                        pdf.set_auto_page_break(auto=True, margin=15)
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        pdf.cell(0, 8, "QuizMolder Export", ln=True)
                        pdf.ln(4)
                        for idx, q in enumerate(quiz, 1):
                            pdf.multi_cell(0, 7, f"Q{idx}. {q['question']}")
                            if q.get("options"):
                                for o in q["options"]:
                                    pdf.multi_cell(0, 7, f"    - {o}")
                            pdf.multi_cell(0, 7, f"Answer: {q['answer']}")
                            pdf.ln(3)
                        # write to temp file and offer download
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        pdf.output(tmp.name)
                        tmp.close()
                        pdf_file = open(tmp.name, "rb")
                        st.download_button("Download PDF", data=pdf_file, file_name=f"quiz_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
                        pdf_file.close()
                        os.unlink(tmp.name)
                    except Exception as e:
                        st.warning(f"PDF export failed: {e}")
                else:
                    st.info("PDF export not available (install `fpdf` to enable).")

    else:
        # When not generated, show last quiz if present
        last = st.session_state.get("last_quiz")
        if last:
            st.info("Showing last generated quiz (click Generate to refresh).")
            for i, q in enumerate(last, 1):
                st.markdown(f"**Q{i}. {q['question']}**")
                if q.get("options"):
                    for opt in q["options"]:
                        st.markdown(f"- {opt}")
                    st.markdown(f"**Answer:** {q['answer']}")
