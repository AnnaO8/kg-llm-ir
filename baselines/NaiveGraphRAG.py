import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from keybert import KeyBERT
import random
from neo4j import GraphDatabase
import ast

load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


def test_connection():
    try:
        with driver.session() as session:
            result = session.run("RETURN 'Neo4j connection successful!' AS message")
            message = result.single()["message"]
            print(f"{message}")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False
    

def check_entities_in_kg(entities):
    found_entities = []
    missing_entities = []
    
    with driver.session() as session:
        for entity in entities:
            query = "MATCH (n) WHERE n.name = $name RETURN n.name"
            result = session.run(query, name=entity)
            records = [record["n.name"] for record in result]
            
            if records:
                found_entities.append(entity)
            else:
                missing_entities.append(entity)

    return found_entities, missing_entities


def BFS(entities):
    paths = set()
    
    with driver.session() as session:
        for entity in entities:
            query = """
            MATCH path = (start {name: $entity})-[*1..2]-(neighbor)
            RETURN 
                [node IN nodes(path) | node.name] AS nodes,
                [rel IN relationships(path) | { 
                    type: rel.type, 
                    start: startNode(rel).name, 
                    end: endNode(rel).name
                }] AS relations
            LIMIT 7
            """
            result = session.run(query, entity=entity)
        
            for record in result:
                nodes = record["nodes"]
                relations = record["relations"]

                full_path = []
                for i in range(len(relations)):
                    full_path.append(nodes[i])
                    
                    relation = relations[i]
                    relation_type = relation["type"]

                    if relation["start"] == nodes[i]:
                        relation_string = f"{relation_type} ->"
                    else:
                        relation_string = f"{relation_type} <-"
                    
                    full_path.append(relation_string)  
                    
                full_path.append(nodes[-1])
                
                
                paths.add(tuple(full_path))

    return [list(p) for p in paths]



def find_paths_between_entities(entities):
    paths = set()
    
    with driver.session() as session:
        query = """
        MATCH path = allShortestPaths((e1)-[*]-(e2)) 
        WHERE e1.name IN $entities AND e2.name IN $entities AND e1 <> e2
        RETURN 
            [node IN nodes(path) | node.name] AS nodes,
            [rel IN relationships(path) | { 
                type: rel.type, 
                start: startNode(rel).name, 
                end: endNode(rel).name
            }] AS relations
        """
        result = session.run(query, entities=entities)
        
        for record in result:
            nodes = record["nodes"]
            relations = record["relations"]

            full_path = []
            for i in range(len(relations)):
                full_path.append(nodes[i])
                
                relation = relations[i]
                relation_type = relation["type"]

                if relation["start"] == nodes[i]:
                    relation_string = f"{relation_type} ->"
                else:
                    relation_string = f"{relation_type} <-"
                
                full_path.append(relation_string)  
                
            full_path.append(nodes[-1])
            

            reversed_path = reverse_path(full_path)


            canonical_path = min(tuple(full_path), reversed_path)
            
            paths.add(canonical_path)

    return [list(p) for p in paths]


def reverse_path(path):
    reversed_path = []
    for i in range(len(path) - 1, 0, -2):  # Gehe rückwärts durch den Pfad
        reversed_path.append(path[i])  # Knoten hinzufügen
        relation = path[i - 1]
        
        # Drehe die Richtung um
        if "->" in relation:
            reversed_relation = relation.replace("->", "<-")
        elif "<-" in relation:
            reversed_relation = relation.replace("<-", "->")
        else:
            print("!!! Error in reverse_path for path: ", path)

        reversed_path.append(reversed_relation)
    
    reversed_path.append(path[0])
    return tuple(reversed_path)



def Naive_GraphRAG(QA_df, output_name):
    print("START Naive_GraphRAG()")
    RAG_df = pd.DataFrame()
    kw_model = KeyBERT()
    print("1 Retrieve Relevant Documents...")
    for i in range(QA_df.shape[0]):
        query = QA_df['Question'].iloc[i]
        
        query = QA_df.iloc[i].Question

        extracted_keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 2), stop_words=None)
        keywords = [kw[0] for kw in extracted_keywords]
        
        found_entities, missing = check_entities_in_kg(keywords)
        
        # perform BFS: surrounding nodes with breadth limit of 7 and each depth until 2
        paths_BFS = BFS(found_entities)
        #print("\nPath: ", paths_BFS)
        
        # perform DFS
        paths_DFS = find_paths_between_entities(found_entities)
        paths = random.sample(paths_DFS, min(7,len(paths_DFS)))
        
        final_paths = []
        seen = set()
        
        # paths_BFS -> final_paths (distinct)
        for path in paths_BFS:
            path_tuple = tuple(path)
            if path_tuple not in seen:
                seen.add(path_tuple)
                final_paths.append(path)
        
        # paths (DFS) -> final_paths (distinct)        
        for path in paths:
            path_tuple = tuple(path)
            if path_tuple not in seen:
                seen.add(path_tuple)
                final_paths.append(path)     
        #print("\nFinal Paths: ",final_paths)           
        
        
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
        
        prompt = prompt_template.format(context=final_paths, query=query)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model="gpt-4o-mini",
        )

        answer = response.choices[0].message.content
        
        
        new_row = {'query':query,'paths':final_paths, 'entities':found_entities, 'missing_entities':missing, 'answer':answer}
        RAG_df = RAG_df._append(new_row, ignore_index=True)
        print(f"\rProgress: {i+1}/{QA_df.shape[0]}", end="", flush=True)
    
    print("\n2 Save result file ", output_name, "...")
    
    file_path_name = f"../evaluation/results/{output_name}.csv"
    RAG_df.to_csv(file_path_name, index=False, encoding='utf-8')
    print("\nEND Naive_GraphRAG()")
    
    
if __name__ == "__main__":
    #kg = pd.read_csv(r"../data/Economic_KG.csv")
    kg = pd.read_csv(r"../data/Economic_KG_withUnderscore.csv")
    URI = "bolt://localhost:7687"
    USERNAME = "neo4j"
    PASSWORD = "875421963"
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    test_connection()


    QA_descriptive = pd.read_csv(r"../data/QA_descriptive.csv")
    QA_singleEntity = pd.read_csv(r"../data/QA_singleEntity.csv")
    QA_yesNo = pd.read_csv(r"../data/QA_yesNo.csv")
    QA_NULL = pd.read_csv(r"../data/QA_NULL.csv")
    
    Naive_GraphRAG(QA_df=QA_descriptive, output_name='Naive_GraphRAG_descriptive')
    Naive_GraphRAG(QA_df=QA_singleEntity, output_name='Naive_GraphRAG_singleEntity')
    Naive_GraphRAG(QA_df=QA_yesNo, output_name='Naive_GraphRAG_yesNo')
    Naive_GraphRAG(QA_df=QA_NULL, output_name='Naive_GraphRAG_NULL')

    '''
    QA_converging = pd.read_csv(r"../data/QA_Converging.csv")
    QA_divergent = pd.read_csv(r"../data/QA_Divergent.csv")
    QA_linear = pd.read_csv(r"../data/QA_Linear.csv")

    
    Naive_GraphRAG(QA_df=QA_converging, output_name='Naive_GraphRAG_Converging')
    Naive_GraphRAG(QA_df=QA_divergent, output_name='Naive_GraphRAG_Divergent')
    Naive_GraphRAG(QA_df=QA_linear, output_name='Naive_GraphRAG_Linear')
    '''