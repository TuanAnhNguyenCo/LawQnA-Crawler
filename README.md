# Project Introduction

This project aims to develop an effective Vietnamese legal question-answering (QA) system using a Retrieval Augmented Generation (RAG) approach. The core objective is to provide accurate and relevant responses to user inquiries regarding legal matters.

# Technical Approach

The system leverages a combination of cutting-edge technologies:

* **Text Embeddings:** OpenAI's `text-embedding-3-small` model is used to generate high-quality embeddings for both user questions and legal documents. These embeddings capture semantic meaning, enabling more precise comparisons between text inputs.
* **Language Model:** The `gpt-4o-mini-2024-07-18` model (from OpenAI) serves as the underlying language model (LLM). This LLM answers question from users
* **Re-ranking with Cohere API:** Due to the current lack of Cross-Encoders specifically designed for Vietnamese, the Cohere API is employed for re-ranking search results. This step enhances the accuracy of the final answer selection.
* **Dynamic Data Crawling:**  If a user's question cannot be answered by the existing stored data, the system automatically crawls relevant legal sources to find the appropriate information. This newly acquired data is then stored along with the answer provided to the user, enriching the knowledge base for future queries.

# Pipeline Overview


