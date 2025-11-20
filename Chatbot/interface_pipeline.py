# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 19:12:36 2025

@author: dayli
"""

# interface_pipeline.py

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI

from config import Config

# Intento de importar RetrievalQA como pide la consigna
try:
    from langchain.chains import RetrievalQA  # type: ignore
    _HAS_LC_RETRIEVALQA = True
except Exception:
    # Si falla (por el problema de langchain_core.pydantic_v1), usamos un fallback
    RetrievalQA = None  # type: ignore
    _HAS_LC_RETRIEVALQA = False


# Fallback: definimos una clase RetrievalQA con from_chain_type
# que imita el comportamiento de langchain.chains.RetrievalQA
if not _HAS_LC_RETRIEVALQA:

    class RetrievalQA:  # type: ignore
        @classmethod
        def from_chain_type(
            cls,
            llm,
            retriever,
            chain_type: str = "stuff",
            return_source_documents: bool = True,
        ):
            """
            Fallback simple que implementa un RAG:
              - usa el retriever (get_relevant_documents o invoke)
              - "stuff" todo el contexto en un único prompt
              - llama al LLM
              - devuelve result + source_documents
            """

            def _invoke(inputs):
                # LangChain normalmente pasa {"query": "..."} o {"question": "..."}
                if isinstance(inputs, dict):
                    question = (
                        inputs.get("query")
                        or inputs.get("question")
                        or str(inputs)
                    )
                else:
                    question = str(inputs)

                # Recuperar documentos relevantes
                if hasattr(retriever, "get_relevant_documents"):
                    docs = retriever.get_relevant_documents(question)
                elif hasattr(retriever, "invoke"):
                    docs = retriever.invoke(question)
                else:
                    docs = []

                # Construir contexto concatenando los chunks
                context = "\n\n".join(doc.page_content for doc in docs)

                prompt = (
                    "You are an assistant that answers questions about the "
                    "software requirements document.\n\n"
                    "Use ONLY the information from the context below.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer in a clear and concise way:"
                )

                # Llamar al LLM de Mistral con manejo de errores (429, etc.)
                try:
                    response = llm.invoke(prompt)
                    answer_text = response.content
                except Exception as e:
                    # Si la API de Mistral falla (por capacidad, clave, red, etc.),
                    # devolvemos un mensaje amigable pero NO rompemos el chatbot.
                    answer_text = (
                        "The Mistral API is currently unavailable or over capacity. "
                        "Please try again in a moment.\n\n"
                        f"(Technical detail: {e})"
                    )

                # Emular el formato de salida de RetrievalQA
                return {
                    "result": answer_text,
                    "source_documents": docs if return_source_documents else None,
                }

            # Devolvemos un objeto tipo "chain" que es invocable y tiene invoke()
            class SimpleRAGChain:
                def __call__(self, inputs):
                    return _invoke(inputs)

                def invoke(self, inputs):
                    return _invoke(inputs)

            return SimpleRAGChain()


def load_vector_store():
    """
    Carga el vector store de Chroma ya creado en el Loading Pipeline.

    Usa:
      - misma colección que Task 3 (Config.COLLECTION_NAME)
      - mismo directorio persistente (Config.CHROMA_DIR)
      - mismo modelo de embeddings 'all-MiniLM-L6-v2'
    """
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name=Config.COLLECTION_NAME,
        persist_directory=Config.CHROMA_DIR,
        embedding_function=embeddings,
    )

    return vector_store


def create_rag_chain():
    """
    Task 4: Crear el RAG chain con Mistral + Retriever.

    a) Crea instancia de ChatMistralAI usando la API key y el modelo del .env
    b) Convierte el vector store a retriever con k = 3
    c) Construye RetrievalQA.from_chain_type con:
          - chain_type="stuff"
          - return_source_documents=True
    Imprime: "RAG chain successfully created"
    """
    # a) LLM de Mistral usando variables del .env (a través de Config)
    mistral_llm = ChatMistralAI(
        api_key=Config.MISTRAL_API_KEY,
        model=Config.MISTRAL_MODEL_NAME,
    )

    # Cargar el vector store que ya construiste en el Loading Pipeline
    vector_store = load_vector_store()

    # b) Crear retriever (top 3 chunks más relevantes)
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    # c) Crear la cadena RAG con RetrievalQA (real o fallback)
    rag_chain = RetrievalQA.from_chain_type(
        llm=mistral_llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

    print("RAG chain successfully created")
    return rag_chain


if __name__ == "__main__":
    # ===== Task 5: Chatbot loop =====

    # a) Mensaje de inicio
    print("Starting Daylin RAG Chatbot setup..")

    # Crear la cadena RAG (Task 4)
    rag_chain = create_rag_chain()

    # b) Mensaje cuando está listo
    print("RAG Chatbot Daylin ready! Type 'exit' to quit.")

    # c) Bucle interactivo
    while True:
        user_query = input("\nYou: ")

        # salir si escribe 'exit'
        if user_query.strip().lower() == "exit":
            break

        # Enviar la pregunta al RAG usando invoke (como pide la consigna)
        response = rag_chain.invoke({"query": user_query})

        # Extraer respuesta y documentos fuente
        if isinstance(response, dict):
            answer = response.get("result", "")
            source_docs = response.get("source_documents", []) or []
        else:
            # por si acaso, pero normalmente será dict
            answer = str(response)
            source_docs = []

        # Imprimir primeros 200 caracteres de cada source document
        print("\n--- Source Chunks (first 200 chars) ---")
        for idx, doc in enumerate(source_docs, start=1):
            snippet = doc.page_content[:200].replace("\n", " ")
            print(f"[{idx}] {snippet}")

        # Mostrar respuesta del bot
        print("\n--- Bot Answer ---")
        print(answer)
