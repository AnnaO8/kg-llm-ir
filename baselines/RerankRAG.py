import os
from openai import OpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def RerankRAG(QA_df, output_name, top_k=16, rerank_k=7):
    print("START RerankRAG():")    
    RAG_df = pd.DataFrame()

    
    
    print("3 Retrieve relevant documents...")
    
    for i in range(QA_df.shape[0]):
        query = QA_df['Question'].iloc[i]

        # Step 4: Retrieve Top-K Relevant Triples
        query_embedding = retrieval_model.encode([query])
        distances, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), top_k)
        retrieved_triples = [triples[idx] for idx in indices[0]]


        # Step 5: Load Re-Ranker Model & Rank Retrieved Triples
        scores = reranker.predict([(query, triple) for triple in retrieved_triples])
        reranked_triples = [triple for _, triple in sorted(zip(scores, retrieved_triples), reverse=True)][:rerank_k]
        reranked_triples_list = []
        for triple in reranked_triples:
            parts = triple.split(" ", 2)  # each triple consists of 3 parts: n, r, m
            if len(parts) == 3:
                reranked_triples_list.append(tuple(parts))
            else:
                print("!!! Error: ", triple.get_content())
        
        
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

        prompt = prompt_template.format(context=reranked_triples, query=query)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model="gpt-4o-mini",
        )

        answer = response.choices[0].message.content

        new_row = {'query': query, 'first retrieved triples': retrieved_triples, 'reranked triples': reranked_triples_list, 'answer': answer}
        RAG_df = RAG_df._append(new_row, ignore_index=True)
        print(f"\rProgress: {i+1}/{QA_df.shape[0]}", end="", flush=True)
        
    print("\n3 Save result file ", output_name, "...")
    file_path_name = f"../evaluation/results/{output_name}.csv"
    RAG_df.to_csv(file_path_name, index=False, encoding='utf-8')
    print("\nEND RR_RAG()")
    
    
    
if __name__ == "__main__":
    kg = pd.read_csv(r"../data/Economic_KG_withUnderscore.csv")
    
    triples = [f"{row['n']} {row['r']} {row['m']}" for _, row in kg.iterrows()]
    
    # Step 2: Load Embedding Model and Encode KG Triples
    print("1 Load Embedding Model and encode KG triples...")
    retrieval_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    triple_embeddings = retrieval_model.encode(triples)

    # Step 3: Create FAISS Index for Retrieval
    print("2 Create FAISS Index for Retrieval...")
    dimension = triple_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(triple_embeddings, dtype=np.float32))
    
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    QA_descriptive = pd.read_csv(r"../data/QA_descriptive.csv")
    QA_singleEntity = pd.read_csv(r"../data/QA_singleEntity.csv")
    QA_yesNo = pd.read_csv(r"../data/QA_yesNo.csv")
    QA_NULL = pd.read_csv(r"../data/QA_NULL.csv")
    
    RerankRAG(QA_df=QA_descriptive, output_name='RerankRAG_descriptive')
    RerankRAG(QA_df=QA_singleEntity, output_name='RerankRAG_singleEntity')
    RerankRAG(QA_df=QA_yesNo, output_name='RerankRAG_yesNo')
    RerankRAG(QA_df=QA_NULL, output_name='RerankRAG_NULL')
    
    ''' 
    QA_converging = pd.read_csv(r"../data/QA_Converging.csv")
    QA_divergent = pd.read_csv(r"../data/QA_Divergent.csv")
    QA_linear = pd.read_csv(r"../data/QA_Linear.csv")
    
    RerankRAG(QA_df=QA_converging, output_name='RerankRAG_Converging')
    RerankRAG(QA_df=QA_divergent, output_name='RerankRAG_Divergent')
    RerankRAG(QA_df=QA_linear, output_name='RerankRAG_Linear')
    '''