from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

def build_eval_dataset(questions: list[str], ground_truths: list[str]) -> dict:
    answers, contexts = [], []
    for q in questions:
        raw = hybrid_retriever.invoke(q)
        top = rerank(q, raw)
        context_text = "\n\n".join(d.page_content for d in top)
        # Get LLM answer (call your FastAPI or run inline)
        answer = get_llm_answer(q, context_text)
        answers.append(answer)
        contexts.append([d.page_content for d in top])
    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

# Run eval
data = build_eval_dataset(questions, ground_truths)
result = evaluate(
    Dataset.from_dict(data),
    metrics=[faithfulness, answer_relevancy, context_recall],
)

print(result)
# Example output you want in your README:
# faithfulness: 0.91 | answer_relevancy: 0.88 | context_recall: 0.84