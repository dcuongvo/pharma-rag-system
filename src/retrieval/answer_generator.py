# from src.utils.llm_factory import get_llm
# from src.utils.llm_utils import safe_llm_call

# class AnswerGenerator:
#     def __init__(self, provider=None, model=None):
#         self.llm = get_llm(provider=provider, model=model)

#     def generate(self, query: str, chunks):
#         context = "\n\n".join(
#             f"[Chunk {i+1}]\n{chunk.text}"
#             for i, (chunk, _) in enumerate(chunks)
#         )

#         prompt = f"""
#             You are a pharmaceutical document assistant.

#             Answer the question using ONLY the context below.
#             If the answer is not found, say "No reliable answer found."

#             Question:
#             {query}

#             Context:
#             {context}

#             Answer:
#         """

#         response = safe_llm_call(self.llm, prompt)
#         return response.text.strip()
 


from src.utils.llm_factory import get_llm
from src.utils.llm_utils import safe_llm_call


class AnswerGenerator:
    def __init__(self, provider=None, model=None):
        self.llm = get_llm(provider=provider, model=model)

    def generate(self, query: str, chunks):
        context = "\n\n".join(
            f"[Chunk {i+1}]\n{chunk.text}"
            for i, (chunk, _) in enumerate(chunks)
        )

        prompt = f"""
        You are a pharmaceutical document assistant.

        Answer the question using ONLY the provided context.

        Rules:
        - If there is one clear answer, return that single answer.
        - If the question is broad or ambiguous and multiple relevant answers exist, return only the top 3 to 5 most relevant answers.
        - Do NOT dump every possible value from the context.
        - Prefer the most relevant values from the highest-ranked chunks.
        - If the answer is not clearly supported, say: "No reliable answer found."
        - Be concise.

        Question:
        {query}

        Context:
        {context}

        Response format:
        - For one clear answer:
        Answer: <single answer>

        - For multiple relevant answers:
        Answer:
        - <answer 1>
        - <answer 2>
        - <answer 3>

        Answer:
        """
        response = safe_llm_call(self.llm, prompt)
        return response.text.strip()