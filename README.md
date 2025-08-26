# QuizMolder – Offline AI Quiz Generator

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**QuizMolder** helps students and teachers automatically generate quizzes from lecture notes.  
It is completely free, offline-first, and open-source. No paid APIs, no constant internet required.

---

## Features
- Generate MCQs directly from notes  
- Works offline with lightweight NLP  
- Simple and accessible interface  
- Local storage for privacy  

---

## Tech Stack
- Python  
- NLTK (text processing)  
- Streamlit (web interface)  
- SQLite/CSV (data storage)  

---

## Installation & Setup

Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/quizmolder.git
cd quizmolder
pip install -r requirements.txt
````

Run the app:

```bash
streamlit run app.py
```

---

## Example

**Input Notes**

```
The Earth revolves around the Sun.
The Moon revolves around the Earth.
Water boils at 100°C.
```

**Generated Output**

```
Q1. The Earth revolves around the _____.
- Moon
- Earth
- Sun
- Star
Answer: Sun
```

---

## Challenges Faced

* Generating accurate questions without large models
* Optimizing NLP for low-resource machines
* Designing an easy-to-use interface for educators

---

## Hackathon Track

**EdTech & Learning** – AI tutors, gamified learning, and upskilling tools.

---

## License

This project is released under the [MIT License](LICENSE).
