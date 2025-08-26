import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import random

nltk.download('punkt', quiet=True)

def generate_quiz(text, num_questions=5):
    sentences = sent_tokenize(text)
    questions = []

    for i in range(min(num_questions, len(sentences))):
        sent = sentences[i]
        words = word_tokenize(sent)

        # pick a random word as the "blank"
        content_words = [w for w in words if w.isalpha() and len(w) > 3]
        if not content_words:
            continue
        answer = random.choice(content_words)
        question = sent.replace(answer, "_____")

        # create fake options (random words from text)
        all_words = [w for w in word_tokenize(text) if w.isalpha()]
        options = random.sample(all_words, 3)
        options.append(answer)
        random.shuffle(options)

        questions.append({
            "question": question,
            "answer": answer,
            "options": options
        })

    return questions
