import os
from openai import OpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
import pandas as pd
from llama_index.core.schema import Document
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer


load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


def HybridRAG(QA_df, output_name):
    sparse_weight = 0.5
    dense_weight = 0.5
    top_k = 5
    print("START HybridRAG():")
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
    
    
    # Convert triples to text format for LlamaIndex
    documents = [
        Document(text=f"{row['n']} {row['r']} {row['m']}")
        for _, row in kg.iterrows()
    ]
    
    # Parse documents into nodes
    splitter = SentenceSplitter(chunk_size=512)
    nodes = splitter.get_nodes_from_documents(documents)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes)


    RAG_df = pd.DataFrame()
    print("2 Retrieve relevant documents...")
    for i in range( QA_df.shape[0]):
        query = QA_df['Question'].iloc[i]

        # Sparse retrieval (BM25)
        sparse_results = bm25_retriever.retrieve(query)
        sparse_triples = []
        for node in sparse_results:
            parts = node.get_content().split(" ", 2)  # Maximal 3 Teile: n, r, m
            if len(parts) == 3:
                sparse_triples.append(tuple(parts))

        # Dense retrieval 
        query_embedding = embedding_model.encode([query])
        _, indices = index.search(query_embedding, top_k)


        dense_context = [triples_tuple_struc[i] for i in indices[0]]
        dense_triples = dense_context

        combined_context = list(set(dense_triples + sparse_triples))
        
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

        prompt = prompt_template.format(context=combined_context, query=query)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model="gpt-4o-mini",
        )

        answer = response.choices[0].message.content

        new_row = {'query':query,'sparse context':sparse_triples, 'dense context': dense_context, 'combined context': combined_context, 'answer': answer}
        RAG_df = RAG_df._append(new_row, ignore_index=True)
        print(f"\rProgress: {i+1}/{QA_df.shape[0]}", end="", flush=True)

    print("\n3 Save result file ", output_name, "...")
    file_path_name = f"../evaluation/results/{output_name}.csv"
    RAG_df.to_csv(file_path_name, index=False, encoding='utf-8')
    print("\nEND HybridRAG()")
    
    
if __name__ == "__main__":



    QA_descriptive = pd.read_csv(r"../data/QA_descriptive.csv")
    QA_singleEntity = pd.read_csv(r"../data/QA_singleEntity.csv")
    QA_yesNo = pd.read_csv(r"../data/QA_yesNo.csv")
    QA_NULL = pd.read_csv(r"../data/QA_NULL.csv")
    
    HybridRAG(QA_df=QA_descriptive, output_name='HybridRAG_descriptive')
    HybridRAG(QA_df=QA_singleEntity, output_name='HybridRAG_singleEntity')
    HybridRAG(QA_df=QA_yesNo, output_name='HybridRAG_yesNo')
    HybridRAG(QA_df=QA_NULL, output_name='HybridRAG_NULL')

    ''' 
    QA_converging = pd.read_csv(r"../data/QA_Converging.csv")
    QA_divergent = pd.read_csv(r"../data/QA_Divergent.csv")
    QA_linear = pd.read_csv(r"../data/QA_Linear.csv")

    
    HybridRAG(QA_df=QA_converging, output_name='HybridRAG_Converging')
    HybridRAG(QA_df=QA_divergent, output_name='HybridRAG_Divergent')
    HybridRAG(QA_df=QA_linear, output_name='HybridRAG_Linear')
    '''
    