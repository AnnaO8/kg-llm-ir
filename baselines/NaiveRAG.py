import os
from openai import OpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def NaiveRAG(QA_df, output_name):
    print("START NaiveRAG():")

    RAG_df = pd.DataFrame()

    print("2 Retrieve relevant documents...")
    for i in range(QA_df.shape[0]):
    #for i in range(5):

        query = QA_df.iloc[i].Question
        # Step 1: Encode the query and retrieve top-k relevant documents
        query_embedding = embedding_model.encode([query])
        k = 6  # Number of documents to retrieve
        _, indices = index.search(query_embedding, k)

        # Retrieve the top-k relevant documents
        #retrieved_docs = [knowledge_base[i] for i in indices[0]]
        retrieved_docs = [triples_tuple_struc[i] for i in indices[0]]

        # Step 2: Generate an answer using OpenAI's GPT model
        #retrieved_context = "\n".join(retrieved_docs)
        retrieved_context = retrieved_docs

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
         
        new_row = {'query': query, 'Retrieved Document': retrieved_context, 'answer': answer}
        RAG_df = RAG_df._append(new_row, ignore_index=True)
        print(f"\rProgress: {i+1}/{QA_df.shape[0]}", end="", flush=True)
    print("\n3 Save result file ", output_name, "...")
    file_path_name = f"../evaluation/results/{output_name}.csv"
    RAG_df.to_csv(file_path_name, index=False, encoding='utf-8')
    print("\nEND NaiveRAG()")
    
    
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
    
    
    ''' 
    QA_converging = pd.read_csv(r"../data/QA_Converging.csv")
    QA_divergent = pd.read_csv(r"../data/QA_Divergent.csv")
    QA_linear = pd.read_csv(r"../data/QA_Linear.csv")
    
    NaiveRAG(QA_df=QA_converging, output_name='NaiveRAG_Converging')
    NaiveRAG(QA_df=QA_divergent, output_name='NaiveRAG_Divergent')
    NaiveRAG(QA_df=QA_linear, output_name='NaiveRAG_Linear')
    '''
    
    QA_descriptive = pd.read_csv(r"../data/QA_descriptive.csv")
    QA_singleEntity = pd.read_csv(r"../data/QA_singleEntity.csv")
    QA_yesNo = pd.read_csv(r"../data/QA_yesNo.csv")
    QA_NULL = pd.read_csv(r"../data/QA_NULL.csv")
    NaiveRAG(QA_df=QA_descriptive, output_name='NaiveRAG_descriptive')
    NaiveRAG(QA_df=QA_singleEntity, output_name='NaiveRAG_singleEntity')
    NaiveRAG(QA_df=QA_yesNo, output_name='NaiveRAG_yesNo')
    NaiveRAG(QA_df=QA_NULL, output_name='NaiveRAG_NULL')
    