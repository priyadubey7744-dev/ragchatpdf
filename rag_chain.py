from groq import Groq
from src.vector_store import VectorStore


class RAGChain:
    """
    Core RAG pipeline: Retrieve → Augment → Generate

    Flow:
    1. Take user question
    2. Search vector store for relevant chunks (Retrieval)
    3. Inject chunks into LLM prompt (Augmentation)
    4. LLM generates grounded answer (Generation)
    """

    def __init__(self, api_key: str, vector_store: VectorStore):
        self.client = Groq(api_key=api_key)
        self.vector_store = vector_store
        self.model = "llama3-8b-8192"

    def answer(self, question: str, chat_history: list = None) -> tuple[str, list[str]]:
        """
        Generate an answer grounded in retrieved document chunks.

        Args:
            question: User's question
            chat_history: Previous messages for context

        Returns:
            Tuple of (answer string, list of source chunks used)
        """
        # Step 1: RETRIEVE relevant chunks
        sources = self.vector_store.search(question, top_k=4)

        if not sources:
            return "I couldn't find relevant information in the document for that question.", []

        # Step 2: AUGMENT — build context from retrieved chunks
        context = "\n\n---\n\n".join(sources)

        # Step 3: Build conversation history (last 4 turns for context)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions ONLY based on the provided document context.\n"
                    "If the answer is not in the context, say: 'This information is not available in the document.'\n"
                    "Be concise, accurate, and cite which part of the document supports your answer when possible.\n\n"
                    f"DOCUMENT CONTEXT:\n{context}"
                )
            }
        ]

        # Add last 4 messages of chat history
        if chat_history:
            for msg in chat_history[-4:]:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

        # Add current question
        messages.append({"role": "user", "content": question})

        # Step 4: GENERATE answer
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=700,
            temperature=0.3  # Lower = more factual, less creative
        )

        answer = response.choices[0].message.content.strip()
        return answer, sources
