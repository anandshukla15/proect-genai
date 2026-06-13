"""
Step 6: ask 5 questions and verify answers are grounded, not hallucinated.

Requires an API key in .env (GROQ_API_KEY or GOOGLE_API_KEY).

    python test_questions.py            # uses sample/notes.pdf
    python test_questions.py my.pdf     # uses your own PDF (edit the questions)
"""

import sys
import warnings

warnings.filterwarnings("ignore")

from rag_chatbot import build_vectorstore, build_qa_chain, ask, NOT_FOUND

PDF = sys.argv[1] if len(sys.argv) > 1 else "sample/notes.pdf"

# 3 questions answerable from sample/notes.pdf + 2 that are NOT in it.
# 'grounded' = the expected outcome we check against.
QUESTIONS = [
    ("What is the chemical equation for photosynthesis?", "in-doc"),
    ("Where does the Calvin cycle take place?", "in-doc"),
    ("Name three factors that affect the rate of photosynthesis.", "in-doc"),
    ("What is the capital of France?", "not-found"),
    ("Who won the 2022 FIFA World Cup?", "not-found"),
]


def main() -> None:
    print(f"Indexing {PDF} ...")
    chain = build_qa_chain(build_vectorstore(PDF))
    print("Ready.\n" + "=" * 70)

    passed = 0
    for question, expected in QUESTIONS:
        out = ask(chain, question)
        answer = out["answer"]
        said_not_found = NOT_FOUND.lower() in answer.lower()

        if expected == "not-found":
            ok = said_not_found
        else:
            ok = not said_not_found

        passed += ok
        print(f"\nQ: {question}")
        print(f"A: {answer}")
        if not said_not_found:
            pages = sorted({d.metadata.get("page", "?") for d in out["sources"]})
            print(f"   sources: pages {pages}")
        print(f"   expected: {expected:9} -> {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 70)
    print(f"Result: {passed}/{len(QUESTIONS)} grounded as expected.")


if __name__ == "__main__":
    main()
