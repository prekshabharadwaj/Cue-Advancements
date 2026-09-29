"""
ClueLy - Conversational QA Chain (LangChain v1 / LCEL)
Builds a retrieval-augmented chat chain backed by FAISS + OpenAI.

Usage:
    from chain.qa_chain import build_qa_chain
    from chain.vectorstore import VectorStoreManager

    vs = VectorStoreManager()
    vs.load()  # or vs.build(docs)

    qa = build_qa_chain(vs)
    result = qa.invoke(
        {"question": "What does the document say about X?"},
        config={"configurable": {"session_id": "default"}},
    )
    print(result)  # str answer
"""

import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

from chain.memory import get_session_history
from chain.vectorstore import VectorStoreManager


_SYSTEM_PROMPT = """\
You are ClueLy, a helpful AI assistant.
Use the following retrieved context to answer the user's question.
If the context does not contain enough information, say so honestly
but still try to be helpful using your own knowledge.

--- Context ---
{context}
--- End context ---
"""


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(
    vs_manager: VectorStoreManager,
    model: str | None = None,
    temperature: float = 0.2,
    retrieval_k: int = 4,
):
    """
    Return a RunnableWithMessageHistory chain that accepts
    {"question": str} and returns a str answer.

    Args:
        vs_manager:   Initialised VectorStoreManager (must be ready).
        model:        OpenAI chat model name (env: OPENAI_CHAT_MODEL).
        temperature:  LLM temperature (0 = deterministic).
        retrieval_k:  Number of context chunks to retrieve per query.
    """
    if not vs_manager.is_ready:
        raise RuntimeError(
            "VectorStoreManager is not initialised. "
            "Call vs_manager.load() or vs_manager.build(docs) first."
        )

    chat_model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    llm = ChatOpenAI(
        model=chat_model,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    retriever = vs_manager.as_retriever(k=retrieval_k)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Core LCEL chain: retrieve → format → prompt → LLM → parse
    rag_chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: _format_docs(
                retriever.invoke(x["question"])
            ))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrap with per-session chat history
    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain_with_history
