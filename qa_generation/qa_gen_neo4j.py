from neo4j import GraphDatabase
import re
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from pydantic import BaseModel
import random

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

def print_path(paths):
    for idx, path in enumerate(paths, start=1):
        print(f"Path {idx}: {path}")


def get_top_degree_nodes(k=10):
    query = """
    MATCH (n)
    RETURN n, COUNT { (n)--() } AS degree
    ORDER BY degree DESC
    LIMIT $limit
    """
    
    with driver.session() as session:
        results = session.run(query, limit=k)
        nodes = [(record["n"].get("name", "Unnamed Node"), record["degree"]) for record in results]
    
    return nodes


def find_k_length_paths(start_node, length = 3):
    query = f"""
    MATCH (startNode {{name: $start_node}})
    MATCH path = (startNode)-[*{length}]->(endNode)
    WHERE NOT any(n IN nodes(path)[1..] WHERE n = startNode)
    RETURN nodes(path) AS entities, relationships(path) AS relations
    """

    with driver.session() as session:
        results = session.run(query, start_node=start_node)
        paths = []
        
        for record in results:
            entities = [node["name"] for node in record["entities"]]
            relations = [rel["type"] for rel in record["relations"]]
            
            path_str = f"{entities[0]}"
            for i in range(len(relations)):
                path_str += f" -[{relations[i]}]-> {entities[i+1]}"
            
            paths.append(path_str)
    
    return paths



class verifyPath(BaseModel):
    meaningful: bool
    explanation: str

class generateQA(BaseModel):
    question: str
    answer: str

def verifyPath_prompt(path):
    prompt = (
        f"The following path is provided: {path}."
    )
    
    completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    temperature=0,
    store=True,
    messages=[
        {"role": "system", "content": "Verify if the following path makes logical sense and is meaningful. First, respond with either 'True' or 'False' to indicate whether the path is logical and meaningful. Afterward, provide a brief explanation for your decision. Analyze the provided path as a whole, considering how the entities and relationships are connected."},
        {"role": "user", "content": prompt}
    ],
    response_format=verifyPath
    )
    return completion


def generateQApairs_prompt_descriptive(path):
    prompt = (
        f"The following path is provided: {path}. "
        "Using only this path, generate a meaningful question and answer pair that reflects its relationships and context."
    )
    
    completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    temperature=0,
    store=True,
    messages=[
        {"role": "system", "content": "Analyze the provided path as a whole, considering how the entities and relationships are connected. Use only the information in the path to generate a meaningful question and answer pair that captures the context and logic of the path. Avoid introducing external knowledge or assumptions."},
        {"role": "user", "content": prompt}
    ],
    response_format=generateQA
    )
    return completion


def generateQApairs_prompt_entity(path):
    prompt = (
        f"The following path is provided: {path}. "
        "Using only this path, generate a meaningful question and answer pair that reflects its relationships and context."
    )
    completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    temperature=0,
    store=True,
    messages=[
        {"role": "system", "content": "Analyze the provided path as a whole, considering how the entities and relationships are connected. Use only the information in the path to generate a meaningful question and answer pair that captures the context and logic of the path. Ensure that the answer is a single entity from the path without introducing external knowledge or assumptions."},
        {"role": "user", "content": prompt}
    ],
    response_format=generateQA
    )
    return completion


def generateQApairs_prompt_yes(path):
    prompt = (
        f"The following path is provided: {path}. "
        "Using only this path, generate a meaningful yes/no question where the answer is 'Yes'."
    )
    completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    temperature=0,
    store=True,
    messages=[
        {"role": "system", "content": "Analyze the provided path as a whole, considering how the entities and relationships are connected. Use only the information in the path to generate a meaningful yes/no question based strictly on the path, ensuring that the correct answer is 'yes'. Avoid introducing external knowledge or assumptions."},
        {"role": "user", "content": prompt}
    ],
    response_format=generateQA
    )
    return completion

def generateQApairs_prompt_not(path):
    prompt = (
        f"The following path is provided: {path}. "
        "Using only this path, generate a meaningful yes/no question where the answer is 'No'."
    )
    completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    temperature=0,
    store=True,
    messages=[
        {"role": "system", "content": "Analyze the provided path as a whole, considering how the entities and relationships are connected. Use only the information in the path to generate a meaningful yes/no question based strictly on the path, ensuring that the correct answer is 'yes'. Avoid introducing external knowledge or assumptions."},
        {"role": "user", "content": prompt}
    ],
    response_format=generateQA
    )
    return completion


def path_pipeline(path, method):
    verification_result = verifyPath_prompt(path)
    verification_data = verification_result.choices[0].message.parsed
    if verification_data.meaningful is True:
        print("\n", "Path ", path, " is meaningful.", "\n")
        method_mapping = {
            'descriptive': generateQApairs_prompt_descriptive,
            'entity': generateQApairs_prompt_entity,
            'yes': generateQApairs_prompt_yes,
            'not': generateQApairs_prompt_not
        }
        qa_function = method_mapping.get(method)
        if qa_function:
            qa_result = qa_function(path)
            qa_data = qa_result.choices[0].message.parsed
            print("Question: ", qa_data.question)
            print("Answer: ", qa_data.answer)
            print("--------------------------------------------------------------------")
            return path, verification_data, qa_data
        else:
            print("!!! Invalid Method !!!")
            return
    else:
        print("Path ", path, " is NOT meaningful.")
        print("--------------------------------------------------------------------")
        return path, verification_data, None


def process_paths_descriptive(paths, QA_df, limit = 5):
    counter = 1
    for path in paths:
        if counter > limit:
            break
        path, verification_data, qa_data = path_pipeline(path, method='descriptive')
        if verification_data.meaningful is True:
            new_row = {'path': path, 'meaningful': verification_data, 'question': qa_data.question, 'answer': qa_data.answer}
            QA_df = pd.concat([QA_df, pd.DataFrame([new_row])], ignore_index=True)
            counter += 1
    return QA_df


def process_paths_entity(paths, QA_df, limit = 5):
    counter = 1
    for path in paths:
        if counter > limit:
            break
        path, verification_data, qa_data = path_pipeline(path, method='entity')
        if verification_data.meaningful is True:
            new_row = {'path': path, 'meaningful': verification_data, 'question': qa_data.question, 'answer': qa_data.answer}
            QA_df = pd.concat([QA_df, pd.DataFrame([new_row])], ignore_index=True)
            counter += 1
    return QA_df


def find_k_length_paths_with_NOT(length=3):
    query = f"""
    MATCH path = ()-[*{length}]->()
    WHERE any(rel IN relationships(path) WHERE toLower(rel.type) CONTAINS "not")
    RETURN nodes(path) AS entities, relationships(path) AS relations
    """

    with driver.session() as session:
        results = session.run(query)
        paths = []
        
        for record in results:
            entities = [node["name"] for node in record["entities"]]
            relations = [rel["type"] for rel in record["relations"]]
            
            path_str = f"{entities[0]}"
            for i in range(len(relations)):
                path_str += f" -[{relations[i]}]-> {entities[i+1]}"
            
            paths.append(path_str)
    
    return paths


def process_paths_yes(paths, QA_df, limit = 3):
    counter = 1
    for path in paths:
        if counter > limit:
            break
        path, verification_data, qa_data = path_pipeline(path, method='yes')
        if verification_data.meaningful is True:
            new_row = {'path': path, 'meaningful': verification_data, 'question': qa_data.question, 'answer': qa_data.answer}
            QA_df = pd.concat([QA_df, pd.DataFrame([new_row])], ignore_index=True)
            counter += 1
    return QA_df


def process_paths_not(paths, QA_df, limit = 2):
    counter = 1
    for path in paths:
        if counter > limit:
            break
        path, verification_data, qa_data = path_pipeline(path, method='not')
        if verification_data.meaningful is True:
            new_row = {'path': path, 'meaningful': verification_data, 'question': qa_data.question, 'answer': qa_data.answer}
            QA_df = pd.concat([QA_df, pd.DataFrame([new_row])], ignore_index=True)
            counter += 1
    return QA_df


if __name__ == '__main__':
    URI = "bolt://localhost:7687"
    USERNAME = "neo4j"
    PASSWORD = "875421963"
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    test_connection()

    #top_10_nodes = get_top_degree_nodes(k=10)
    #print(top_10_nodes)
    '''
    [('euro area', 577), 
    ('Governing Council', 422), 
    ('ECB', 263), 
    ('pandemic', 258), 
    ('firms', 242), 
    ('economic activity', 240), 
    ('inflation', 223), 
    ('review period', 202), 
    ('2022', 202), 
    ('United States', 197)]
    '''
    
    # 22462 results for inflation
    #paths_inflation = find_k_length_paths(start_node='inflation', length=3)
    #print_path(paths_inflation)
    

    paths_euro_area = find_k_length_paths(start_node='euro area', length=3)
    paths_Governing_Council = find_k_length_paths(start_node='Governing Council', length=3)
    paths_ECB = find_k_length_paths(start_node='ECB', length=3) 
    paths_pandemic = find_k_length_paths(start_node='pandemic', length=3)  
    paths_firms = find_k_length_paths(start_node='firms', length=3)
    
    #inflation_20 = random.sample(paths_inflation, 20)
    euro_area_10descr = random.sample(paths_euro_area, 10)
    Governing_Council_10descr = random.sample(paths_Governing_Council, 10)
    ECB_10descr = random.sample(paths_ECB, 10)
    pandemic_10descr = random.sample(paths_pandemic, 10)
    firms_10descr = random.sample(paths_firms, 10)
    
    euro_area_10entity = random.sample(paths_euro_area, 10)
    Governing_Council_10entity = random.sample(paths_Governing_Council, 10)
    ECB_10entity = random.sample(paths_ECB, 10)
    pandemic_10entity = random.sample(paths_pandemic, 10)
    firms_10entity = random.sample(paths_firms, 10)
    
    # ToDo: Check for duplicates in random.sample paths
    
    QA_df = pd.DataFrame()
    
    for path_list in [euro_area_10descr, Governing_Council_10descr, ECB_10descr, pandemic_10descr, firms_10descr]:
        QA_df = process_paths_descriptive(path_list, QA_df)
        
    for path_list in [euro_area_10entity, Governing_Council_10entity, ECB_10entity, pandemic_10entity, firms_10entity]:
        QA_df = process_paths_entity(path_list, QA_df)
        
    
    euro_area_5yes = random.sample(paths_euro_area, 5)
    Governing_Council_5yes = random.sample(paths_Governing_Council, 5)
    ECB_5yes = random.sample(paths_ECB, 5)
    
    for path_list in [euro_area_5yes, Governing_Council_5yes, ECB_5yes]:
        QA_df = process_paths_yes(path_list, QA_df)
    
    
    

    paths_with_not_10 = random.sample(find_k_length_paths_with_NOT(),10)
    QA_df = process_paths_not(paths_with_not_10, QA_df)
    
    QA_df.to_csv('output/QA_dataset_55.csv', index=False, encoding='utf-8')
    
    driver.close()
    