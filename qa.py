# qa.py
from transformers import pipeline

def load_qa():
    # DistilBERT fine-tuned on SQuAD — small & accurate for a mini QA
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def read_context(path="sample_context.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def ask(qa, context: str, question: str):
    """Return answer text, confidence (0..1), and character span."""
    out = qa(question=question, context=context)
    return out["answer"], float(out["score"]), (out["start"], out["end"])

def main():
    print("Loading model (first run downloads weights, later runs use cache)...")
    qa = load_qa()
    context = read_context()
    print(f"\nContext loaded: {len(context)} characters\n"
          "Type your questions below. Type 'exit' to quit.\n")

    while True:
        q = input("Q: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        ans, score, span = ask(qa, context, q)
        print(f"A: {ans}  (confidence: {score:.3f})\n")

if __name__ == "__main__":
    main()
