import streamlit as st
from utils.quizgen.py import generate_quiz

st.title("QuizForge – Offline AI Quiz Generator")
st.write("Paste your notes below and generate quizzes instantly (offline, free).")

# Input text
text_input = st.text_area("Paste your lecture notes here:", height=200)

if st.button("Generate Quiz"):
    if text_input.strip():
        quiz = generate_quiz(text_input)
        st.subheader("Generated Quiz")
        for idx, q in enumerate(quiz, 1):
            st.markdown(f"**Q{idx}. {q['question']}**")
            for opt in q["options"]:
                st.write(f"- {opt}")
            st.write(f"✅ Correct Answer: {q['answer']}")
            st.markdown("---")
    else:
        st.warning("Please paste some text to generate a quiz.")
