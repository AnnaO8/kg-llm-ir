import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from sentence_transformers import CrossEncoder
from neo4j import GraphDatabase


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
    
    
def extract_entities(query):
    system_prompt = """You are an AI that extracts entities relevant for a Knowledge Graph.
    Identify only the main entities from the given user query. 
    Respond with a JSON object containing the extracted entities. Example:
    {
        "entities": ["inflation", "monetary policy", "Governing Council"]
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
    
    extracted_entities = json.loads(response.choices[0].message.content)
    return extracted_entities['entities']
    
    
    
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



def reverse_path(path):
    """ Erstellt die gespiegelte Version eines Pfades mit umgekehrten Relationen. """
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
            reversed_relation = relation  # Falls keine Richtung vorhanden ist

        reversed_path.append(reversed_relation)
    
    reversed_path.append(path[0])  # Letzten Knoten hinzufügen
    return tuple(reversed_path)


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
    
    
    
def format_path_for_scoring(path):
    formatted_parts = []

    for i in range(0, len(path), 2):
        entity = path[i]  # Aktuelle Entität
        
        if i + 1 < len(path):  # Falls eine Beziehung existiert
            relation = path[i + 1]
            next_entity = path[i + 2]  # Zielentität

            # Beziehung mit richtiger Richtung formatieren
            if "->" in relation:
                formatted_parts.append(f"{entity} {relation.replace(' ->', '')} {next_entity}")
            elif "<-" in relation:
                formatted_parts.append(f"{next_entity} {relation.replace(' <-', '')} {entity}")

    return ". ".join(formatted_parts)
    
    
def rank_paths_by_relevance(subquery, paths, top_k=3):
    # Formatiere Pfade für das Scoring-Modell
    formatted_paths = [format_path_for_scoring(path) for path in paths]

    # Erstelle Query-Pfad-Paare für das Modell
    query_path_pairs = [(subquery, path) for path in formatted_paths]
    
    if not query_path_pairs:
        print(f"!!! No Query-Path-Pairs for subquery: {subquery}")
        return []

    # Scoring durch den Cross-Encoder
    scores = cross_encoder.predict(query_path_pairs)
    
    # Sortiere die Pfade nach ihrem Score
    ranked_paths = sorted(zip(scores, paths), key=lambda x: x[0], reverse=True)
    
    # Extrahiere die top-k relevantesten Pfade
    top_paths = [path for _, path in ranked_paths[:top_k]]
    
    return top_paths
    
    

def GraphTRACE(QA_df, output_name):
    print("START GraphTRACE()")
    RAG_df = pd.DataFrame()
    
    print("1 Retrieve and Rank Relevant Documents...")
    for i in range(QA_df.shape[0]):
        query = QA_df['Question'].iloc[i]
        #print("Query: ", query)
        extracted_entities = extract_entities(query)
        #print("Extracted Entities: ", extracted_entities)
        subqueries = generate_subqueries(query)
        #print("Subqueries: ", subqueries)
        found_entities, missing = check_entities_in_kg(extracted_entities)
        #print("Found Entities: ", found_entities)
        #print("Missing Entities: ", missing)
        paths = find_paths_between_entities(found_entities)
        #print("Paths: ", paths)
        
        if not subqueries:
            print(f"NO SUBQUERIES for Query: {query}")
        if not paths:
            print(f"NO PATHS for these found entities: {found_entities}")

        
        top_k = 3
        final_paths = []
        seen = set()
    
        for subquery in subqueries:
            top_paths = rank_paths_by_relevance(subquery, paths, top_k=top_k)
            for path in top_paths:
                path_tuple = tuple(path)
                if path_tuple not in seen:
                    seen.add(path_tuple)
                    final_paths.append(path)
            #print("Top Paths: ", top_paths)
            #print("Final Paths: ", final_paths) 
        
        
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
        #print("Answer: ", answer)
        
        
        new_row = {'query':query,'paths':final_paths, 'subqueries':subqueries, 'entities':found_entities, 'missing_entities':missing, 'answer':answer}
        RAG_df = RAG_df._append(new_row, ignore_index=True)
        print(f"\rProgress: {i+1}/{QA_df.shape[0]}", end="", flush=True)
    
    print("\n3 Save result file ", output_name, "...")
    file_path_name = f"../evaluation/results/{output_name}.csv"
    #RAG_df.to_csv(file_path_name, index=False, encoding='utf-8')
    print("END GraphTRACE()")



if __name__  == "__main__":
    URI = "bolt://localhost:7687"
    USERNAME = "neo4j"
    PASSWORD = "875421963"
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    test_connection()
    
    
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    QA_converging = pd.read_csv(r"../data/QA_Converging.csv")
    QA_divergent = pd.read_csv(r"../data/QA_Divergent.csv")
    QA_linear = pd.read_csv(r"../data/QA_Linear.csv")
    
    
    GraphTRACE(QA_df=QA_converging, output_name='GraphTRACE_Converging')
    GraphTRACE(QA_df=QA_divergent, output_name='GraphTRACE_Divergent')
    GraphTRACE(QA_df=QA_linear, output_name='GraphTRACE_Linear')  
    