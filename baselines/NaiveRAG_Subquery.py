import os
from openai import OpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import json


load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def generate_subqueries(query):
    system_prompt = """You are an AI that generates subqueries from a given user query.
    Break the input query into multiple relevant subqueries that each focus on a specific aspect.
    The subqueries should be self-contained, informative, and avoid redundancy.
    Return the subqueries as a JSON object. Example:
    {
        "subqueries": [
            "What are the main factors influencing inflation?",
            "How does monetary policy respond to inflation?",
            "What is the relationship between the euro area and inflation?"
        ]
    }
    Do not add explanations. Only return valid JSON."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        response_format={"type": "json_object"}
    )

    extracted_subqueries = json.loads(response.choices[0].message.content)
    return extracted_subqueries['subqueries']   


def NaiveRAG_Subquery(QA_df, output_name):
    print("START NaiveRAG_Subquery():")
    
    RAG_df = pd.DataFrame()

    
    print("2 Retrieve relevant documents...")
    for i in range (QA_df.shape[0]):

        query = QA_df.iloc[i].Question
        
        subqueries = generate_subqueries(query)
        retrieved_context = []
        for subquery in subqueries:
            subquery_embedding = embedding_model.encode([subquery])
            k = 2
            _, indices = index.search(subquery_embedding, k)
            #retrieved_docs = [knowledge_base[i] for i in indices[0]]
            retrieved_docs = [triples_tuple_struc[i] for i in indices[0]]
            retrieved_context.extend(retrieved_docs)
            
 

        system_prompt = """You are a helpful assistant specialized in answering questions based strictly on the given context. Do not use any external knowledge."""
        prompt_template = """
                ### Query
                {query}

                ### Context
                The following information has been retrieved to help answer the query. It may consist of triples, paths, or subgraphs extracted from a knowledge graph.

                {context}

                ### Instructions
                Based solely on the context above, provide a concise and accurate answer to the query.  
                Do **not** use any external knowledge or make assumptions beyond what is explicitly stated in the context.  
        """

        prompt = prompt_template.format(context=retrieved_context, query=query)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model="gpt-4o-mini",
        )

        answer = response.choices[0].message.content
        new_row = {'query': query, 'subqueries':subqueries, 'Retrieved Document': retrieved_context, 'answer': answer}
        RAG_df = RAG_df._append(new_row, ignore_index=True)
        print(f"\rProgress: {i+1}/{QA_df.shape[0]}", end="", flush=True)
    
    file_path_name = f"../evaluation/results/{output_name}.csv"
    RAG_df.to_csv(file_path_name, index=False, encoding='utf-8')

    print("\nEND NaiveRAG_Subquery()")
    
    
if __name__ == "__main__":
    #kg = pd.read_csv(r"../data/Economic_KG.csv")
    kg = pd.read_csv(r"../data/Economic_KG_withUnderscore.csv")
    
    # Initialize embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create text representation of triples
    knowledge_base = kg.apply(lambda row: f"{row['n']} {row['r']} {row['m']}", axis=1).tolist()
    
    triples_tuple_struc = kg.apply(lambda row: (row['n'], row['r'], row['m']), axis=1).tolist()

    # Initialize FAISS index
    dimension = 384  # Dimension of the embeddings (specific to the model used)
    index = faiss.IndexFlatL2(dimension)

    # Create embeddings for the knowledge base
    print("1 Create embeddings for the knowledge base...")
    kb_embeddings = embedding_model.encode(knowledge_base)

    # Add embeddings to the FAISS index
    index.add(np.array(kb_embeddings))
    
    
    
    QA_descriptive = pd.read_csv(r"../data/QA_descriptive.csv")
    QA_singleEntity = pd.read_csv(r"../data/QA_singleEntity.csv")
    QA_yesNo = pd.read_csv(r"../data/QA_yesNo.csv")
    QA_NULL = pd.read_csv(r"../data/QA_NULL.csv")
    
    NaiveRAG_Subquery(QA_df=QA_descriptive, output_name='NaiveRAG_Subquery_descriptive')
    NaiveRAG_Subquery(QA_df=QA_singleEntity, output_name='NaiveRAG_Subquery_singleEntity')
    NaiveRAG_Subquery(QA_df=QA_yesNo, output_name='NaiveRAG_Subquery_yesNo')
    NaiveRAG_Subquery(QA_df=QA_NULL, output_name='NaiveRAG_Subquery_NULL')
    
    ''' 
    QA_converging = pd.read_csv(r"../data/QA_Converging.csv")
    QA_divergent = pd.read_csv(r"../data/QA_Divergent.csv")
    QA_linear = pd.read_csv(r"../data/QA_Linear.csv")
    
    NaiveRAG_Subquery(QA_df=QA_converging, output_name='NaiveRAG_Subquery_Converging')
    NaiveRAG_Subquery(QA_df=QA_divergent, output_name='NaiveRAG_Subquery_Divergent')
    NaiveRAG_Subquery(QA_df=QA_linear, output_name='NaiveRAG_Subquery_Linear') 
    '''   
    