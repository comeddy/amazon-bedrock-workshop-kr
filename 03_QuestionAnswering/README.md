# Lab 3 - Question Answering

## Introduction

QA(질문 답변)는 자연어로 제기된 사실적인 쿼리에 대한 답변을 추출하는 중요한 작업입니다. 일반적으로 QA 시스템은 정형 또는 비정형 데이터가 포함된 지식창고에 대한 쿼리를 처리하고 정확한 정보가 포함된 답변을 생성합니다. 특히 엔터프라이즈 사용 사례에서 유용하고 신뢰할 수 있으며 신뢰할 수 있는 질문 답변 시스템을 개발하려면 높은 정확도를 보장하는 것이 핵심입니다.

Amazon Titan, Anthropic Claude, AI21 Jurassic 2와 같은 제너레이티브 AI 모델은 확률 분포를 사용하여 질문에 대한 답변을 생성합니다. 이러한 모델은 방대한 양의 텍스트 데이터를 학습하여 시퀀스에서 다음에 무엇이 나올지 또는 특정 단어 다음에 어떤 단어가 나올지 예측할 수 있습니다. 그러나 데이터에는 항상 어느 정도의 불확실성이 존재하기 때문에 이러한 모델은 모든 질문에 대해 정확하거나 결정적인 답변을 제공할 수는 없습니다.

기업은 도메인별 및 독점 데이터를 쿼리하고 해당 정보를 사용하여 질문에 답해야 하며, 더 일반적으로는 모델이 학습되지 않은 데이터도 사용해야 합니다.

## Patterns

In these labs we will explore two QA patterns:

1. First where questions are sent to the model where by we will get answers based on the base model with no modifications.
This poses a challenge,
outputs are generic to common world information, not specific to a customers specific business, and there is no source of information.

    ![Q&A](./images/51-simple-rag.png)

2. The Second Pattern where we use Retrieval Augmented Generation which improves upon the first where we concatenate our questions with as much relevant context as possible, which is likely to contain the answers or information we are looking for.
The challenge here, There is a limit on how much contextual information can be used is determined by the token limit of the model.
    ![RAG Q&A](./images/52-rag-with-external-data.png)

This can be overcome by using Retrival Augmented Generation (RAG) 

## How Retrieval Augmented Generation (RAG) works

RAG combines the use of embeddings to index the corpus of the documents to build a knowledge base and the use of an LLM to extract the information from a subset of the documents in the knowledge base. 


As a preparation step for RAG, the documents building up the knowledge base are split in chunks of a fixed size (matching the maximum input size of the selected embedding model), and are then passed to the model to obtain the embedding vector. The embedding together with the original chunk of the document and additional metadata are stored in a vector database. The vector database is optimized to efficiently perform similarity search between vectors.

## Target audience
Customers with data stores that may be private or frequently changing. RAG approach solves 2 problems, customers having the following challenges can benefit from this lab.
- Freshness of data: if the data is continously changing and model must only provide latest information.
- Actuality of knowledge: if there is some domain specific knowledge that model might not have understanding of, and the model must output as per the domain data.

## Objective

After this module you should have a good understanding of:

1. What is the QA pattern and how it leverages Retrieval Augmented Generation (RAG)
2. How to use Bedrock to implement a Q&A RAG solution


In this module we will walk you through how to implement the QA pattern with Bedrock. 
Additionally, we have prepared the embeddings to be loaded in the vector database for you. 

Take note you can use Titan Embeddings to obtain the embeddings of the user question, then use those embedding to retrieve the most relevant documents from the vector database, build a prompt concatenating the top 3 documents and invoke the LLM model via Bedrock.

## Notebooks

1. [Q&A with model knowledge and small context](./00_qa_w_bedrock_titan.ipynb)

2. [Q&A with RAG](./01_qa_w_rag_claude.ipynb)