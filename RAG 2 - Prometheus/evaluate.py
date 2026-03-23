"""
evaluate.py — Sub-System 2: Evaluation using RAGAS framework with llama3.2:3b as judge LLM.

Scores each response on:
- Answer Relevancy
- Faithfulness
- Contextual Precision
"""

import os
import json
import time
from config import load_config, Config


def evaluate_results(results: list[dict], config: Config = None) -> list[dict]:
    """
    Evaluate RAG pipeline results based on the configured framework.
    """
    if config is None:
        config = load_config()

    print(f"[Evaluate] Framework: {config.evaluation_framework}")
    print(f"[Evaluate] Judge LLM: {config.judge_llm}")
    print(f"[Evaluate] Evaluating {len(results)} results...")

    if config.evaluation_framework == "prometheus":
        return evaluate_with_prometheus(results, config)
    else:
        # Fallback to existing LLM judge (labeled as RAGAS in some places)
        return evaluate_with_llm_judge(results, config)


def evaluate_with_prometheus(results: list[dict], config: Config) -> list[dict]:
    """
    Evaluate results using Prometheus-style prompts.
    Prometheus focuses on a structured rubric for each metric.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    if config.judge_llm_provider == "ollama":
        from langchain_ollama import ChatOllama
        judge_llm = ChatOllama(
            model=config.judge_llm,
            base_url=config.ollama_base_url,
            temperature=0.0,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        judge_llm = ChatGoogleGenerativeAI(
            model=config.judge_llm,
            google_api_key=api_key,
            temperature=0.0,
        )

    # Prometheus-style Rubrics
    rubrics = {
        "answer_relevancy": """### Task Description:
An instruction (might include an Input dataset), a response to evaluate, and a score rubric which outlines a score scale of 1-5 are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: [feedback] [RESULT] [score]\"
4. Please do not generate any other opening, closing, and explanations.

### Instruction:
How relevant is the generated answer to the question?

### Input:
Question: {question}
Generated Answer: {generated_answer}

### Score Rubric:
[Is the response relevant to the instruction?]
Score 1: The response is completely irrelevant to the instruction.
Score 2: The response is mostly irrelevant, only addressing a small part of the instruction.
Score 3: The response is somewhat relevant, but leaves out major parts of the instruction.
Score 4: The response is mostly relevant, addressing most of the instruction.
Score 5: The response is completely relevant, addressing all parts of the instruction.

### Feedback:""",

        "faithfulness": """### Task Description:
An instruction, a response to evaluate, a reference context, and a score rubric which outlines a score scale of 1-5 are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric.
2. After writing a feedback, write a score that is an integer between 1 and 5.
3. The output format should look as follows: \"Feedback: [feedback] [RESULT] [score]\"

### Instruction:
Is the generated answer faithful to the retrieved context?

### Reference Context:
{context}

### Response to Evaluate:
{generated_answer}

### Score Rubric:
[Is the response faithful to the reference context?]
Score 1: The response is completely unfaithful and contains significant hallucinations.
Score 2: The response is mostly unfaithful, with many claims not supported by the context.
Score 3: The response is somewhat faithful, but contains some hallucinations or unsupported claims.
Score 4: The response is mostly faithful, with almost all claims supported by the context.
Score 5: The response is completely faithful, with all claims strictly supported by the context.

### Feedback:""",

        "contextual_precision": """### Task Description:
An instruction, a retrieved context to evaluate, and a score rubric are given.
1. Write feedback and a score (1-5).
2. Format: \"Feedback: [feedback] [RESULT] [score]\"

### Instruction:
How precise and relevant is the retrieved context for answering the question?

### Question:
{question}

### Retrieved Context:
{context}

### Score Rubric:
Score 1: None of the retrieved context is relevant to the question.
Score 2: Only a very small part of the context is relevant.
Score 3: Some parts of the context are relevant, but most are not.
Score 4: Most of the retrieved context is relevant and useful.
Score 5: All of the retrieved context is highly relevant and directly answers the question.

### Feedback:"""
    }

    from concurrent.futures import ThreadPoolExecutor

    def evaluate_metric(metric, prompt_text, r, context_text):
        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | judge_llm | StrOutputParser()
        try:
            response = chain.invoke({
                "question": r["question"],
                "generated_answer": r["generated_answer"],
                "context": context_text,
            })
            
            import re
            score_match = re.search(r"\[RESULT\]\s*(\d)", response)
            if not score_match:
                score_match = re.search(r"(?:RESULT|Score|score)\D*?(\d)", response, re.IGNORECASE | re.DOTALL)
            
            if not score_match:
                digits = re.findall(r"(\d)", response)
                if digits:
                    score_1to5 = int(digits[-1])
                    if 1 <= score_1to5 <= 5:
                        return metric, (score_1to5 - 1) / 4.0
                return metric, 0.0
            else:
                score_1to5 = int(score_match.group(1))
                return metric, (score_1to5 - 1) / 4.0
        except Exception as e:
            print(f"    [Prometheus] Error in {metric}: {e}")
            return metric, 0.0

    for i, r in enumerate(results):
        print(f"  [Prometheus] Evaluating question {i+1}/{len(results)}...")
        r["evaluation_scores"] = {}
        context_text = "\n---\n".join(r.get("retrieved_context", ["No context"]))

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(evaluate_metric, m, p, r, context_text) for m, p in rubrics.items()]
            for future in futures:
                metric, score = future.result()
                r["evaluation_scores"][metric] = score

        r["evaluation_framework"] = "prometheus"
        # Small delay between questions
        time.sleep(0.5)

    return results


def evaluate_with_llm_judge(results: list[dict], config: Config) -> list[dict]:
    """
    Evaluation using a local Ollama model (llama3.2:3b) as a judge LLM.
    Scores each response on answer relevancy, faithfulness, and contextual precision.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    if config.judge_llm_provider == "ollama":
        from langchain_ollama import ChatOllama
        judge_llm = ChatOllama(
            model=config.judge_llm,
            base_url=config.ollama_base_url,
            temperature=0.0,
            format="json",  # Force JSON output to skip reasoning and speed up evaluation
        )
    else:
        # Fallback to Google if configured
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        judge_llm = ChatGoogleGenerativeAI(
            model=config.judge_llm,
            google_api_key=api_key,
            temperature=0.0,
            max_output_tokens=1024,
            timeout=120,
            max_retries=3,
        )

    eval_prompt = ChatPromptTemplate.from_messages([
        ("human", """You are an expert evaluation judge for a RAG (Retrieval-Augmented Generation) system.
Evaluate the generated answer against the expected answer and retrieved context.

Score each metric from 0.0 to 1.0:

1. **Answer Relevancy**: How relevant is the generated answer to the question?
   - 1.0 = perfectly relevant, directly answers the question
   - 0.0 = completely irrelevant

2. **Faithfulness**: Is the generated answer faithful to the retrieved context (no hallucination)?
   - 1.0 = entirely based on retrieved context
   - 0.0 = completely hallucinated

3. **Contextual Precision**: How precise is the retrieved context for answering the question?
   - 1.0 = all retrieved context is highly relevant
   - 0.0 = none of the retrieved context is relevant

Output ONLY a raw JSON object with these three scores. 
DO NOT include any thinking process, reasoning, or preamble. 
Expected formatting:
{{"answer_relevancy": 0.8, "faithfulness": 0.7, "contextual_precision": 0.9}}

---

Question: {question}

Generated Answer: {generated_answer}

Expected Answer: {expected_answer}

Retrieved Context:
{context}

Scores (JSON only):"""),
    ])

    chain = eval_prompt | judge_llm | StrOutputParser()

    for i, r in enumerate(results):
        print(f"  [Judge] Evaluating question {i+1}/{len(results)}...")
        
        # Manual retry loop for robustness against 429s
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                context_text = "\n---\n".join(r.get("retrieved_context", ["No context"]))
                response = chain.invoke({
                    "question": r["question"],
                    "generated_answer": r["generated_answer"],
                    "expected_answer": r["expected_answer"],
                    "context": context_text,
                })
                
                # Parse JSON scores from response
                import re
                json_match = re.search(r"\{[^}]+\}", response)
                if json_match:
                    scores = json.loads(json_match.group())
                    r["evaluation_scores"] = {
                        "answer_relevancy": round(float(scores.get("answer_relevancy", 0.0)), 4),
                        "faithfulness": round(float(scores.get("faithfulness", 0.0)), 4),
                        "contextual_precision": round(float(scores.get("contextual_precision", 0.0)), 4),
                    }
                else:
                    r["evaluation_scores"] = {"answer_relevancy": 0.0, "faithfulness": 0.0, "contextual_precision": 0.0}
                
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_attempts - 1:
                    time.sleep((attempt + 1) * 2)
                else:
                    print(f"  [Judge] Error evaluating question {i+1}: {e}")
                    r["evaluation_scores"] = {"answer_relevancy": 0.0, "faithfulness": 0.0, "contextual_precision": 0.0}
                    break

        r["evaluation_framework"] = config.evaluation_framework
        # Small delay between calls to let the model "breathe"
        time.sleep(1)

    return results


if __name__ == "__main__":
    # Test with sample data
    sample_results = [{
        "question": "What three new benchmarks were introduced in 2023?",
        "generated_answer": "MMMU, GPQA, and SWE-bench.",
        "expected_answer": "MMMU, GPQA, and SWE-bench.",
        "retrieved_context": ["In 2023, researchers introduced new benchmarks — MMMU, GPQA, and SWE-bench."],
    }]
    config = load_config()
    evaluated = evaluate_results(sample_results, config)
    print(json.dumps(evaluated, indent=2))
