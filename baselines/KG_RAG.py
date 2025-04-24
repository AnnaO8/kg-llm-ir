import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
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
    

def plan_exploration_steps(query):
    prompt = f"""You are an AI assistant planning a step-by-step retrieval from a knowledge graph.

    Given a natural language question, your task is to generate a numbered list of 3–5 **executable traversal steps** through a knowledge graph.
    
    Each step must follow one of these patterns:
    - Identify a node or entity in the graph
    - Find a relationship from a node to another node
    - Filtering, validating or selecting entities

    Each step must be **executable as a graph query**, like: MATCH (n)-[:RELATION]->(m)

    Return the steps as a JSON object. Example:
    Query: "Which former husband of Elizabeth Taylor died in Chichester?"
    Response:
    {{
        "steps": [
            "1. Identify the main subject (e.g., Elizabeth Taylor)",
            "2. Find entities related via was married to",
            "3. From those, retrieve place of death",
            "4. Check if any died in Chichester"
        ]
    }}
    Only return valid JSON."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=1,
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": query}
        ],
        response_format={"type": "json_object"}
    )
    extracted_steps = json.loads(response.choices[0].message.content)
    return extracted_steps['steps']    


# check wether step's goal is to find node(s)/entities
def is_node_exploration(step):
    return any(kw in step.lower() for kw in ["identify", "find entity", "starting node", "start at"])


def is_relationship_exploration(step):
    return any(kw in step.lower() for kw in ["relation", "relationship", "connected", "via", "attribute"])  

def vectorDB_search(step_text, top_k=5):
    step_embedding = model.encode([step_text])
    D, I = index.search(step_embedding, top_k)
    return [index_to_node[i] for i in I[0]]

def select_relevant_nodes_with_llm(step, candidates):
    options = "\n".join([f"- {c}" for c in candidates])
    
    prompt = f"""Step: {step}
    Here is a list of node candidates:
    {options}
    
    Which of these nodes are most relevant to this step?
    Return only the relevant node(s) as a JSON list with this exact format:
    {{ "relevant_nodes": ["node1", "node2"] }}"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
        )
    
    selected = json.loads(response.choices[0].message.content)
    return selected
    
    
def cypher_query_explorable_relationships(current_nodes):
    #print("cypher_query_explorable_relationships(current_nodes) - current nodes: ", current_nodes)
    # current nodes:  {'relevant_nodes': ['price effects', 'economic relationships', 'impact on consumer prices']}
    rel_types = set()
    with driver.session() as session:
        for node in current_nodes['relevant_nodes']:
            #print("Node: ", node)
            query = """
            MATCH (n)-[r]->()
            WHERE toLower(n.name) = toLower($node)
            RETURN DISTINCT r.type AS rel
            """
            result = session.run(query, node=node)
            for record in result:
                rel_types.add(record["rel"])
    return list(rel_types)  
    

def vector_similarity_rank(explorable_rels, step):
    rel_embeddings = model.encode(explorable_rels)
    step_embedding = model.encode([step])[0]
    
    similarities = [
        (rel, float(np.dot(step_embedding, rel_emb)))  # cosine similarity approx
        for rel, rel_emb in zip(explorable_rels, rel_embeddings)
    ]
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [rel for rel, _ in similarities]


def select_relevant_rels_with_llm(step, candidates):
    options = "\n".join([f"- {c}" for c in candidates])
    
    prompt = f"""Step: {step}
    Here is a list of relationship candidates:
    {options}
    
    Which of these relationships are most relevant to this step?
    Return only the relevant relationships(s) as a JSON list with this exact format:
    {{ "relevant_relationships": ["relationship1", "relationship2"] }}"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
        )
    
    selected = json.loads(response.choices[0].message.content)
    return selected


def get_target_nodes(current_nodes, selected_rels):
    triple = []
    with driver.session() as session:
        for node in current_nodes['relevant_nodes']:
            for rel in selected_rels['relevant_relationships']:
                query = """
                MATCH (n)-[r]->(m)
                WHERE toLower(n.name) = toLower($node)
                  AND r.type = $rel
                RETURN DISTINCT n.name AS source, r.type AS relation, m.name AS target
                """
                result = session.run(query, node=node, rel=rel)
                for record in result:
                    triple.append({
                        "source": record["source"],
                        "relation": record["relation"],
                        "target": record["target"]
                    })
    return triple
    

def eval_state_with_llm(path_traveled, query):
    steps_str = "\n".join(
    f"{step['step']} -> " + ", ".join(step["nodes"]["relevant_nodes"])
    if "nodes" in step else
    f"{step['step']} ->\n" + "\n".join(
        f"    {p['source']} -[{p['relation']}]-> {p['target']}"
        for p in step["paths"]
    )
    for step in path_traveled
    if "nodes" in step or "paths" in step
    )
    '''
    steps_str = "\n".join([
    f"{step['step']} -> " + (
        ", ".join(step.get("nodes", {}).get("relevant_nodes", []))
        if "nodes" in step else
        "\n    " + "\n    ".join(
            f"{p['source']} -[{p['relation']}]-> {p['target']}" 
            for p in step.get("paths", [])
    for step in path_traveled
    ])
    '''
    #print("steps_str: ", steps_str)

    prompt = f"""You are evaluating a knowledge graph traversal.

                Question: "{query}"

                Here is the path traveled so far:
                {steps_str}

                Decide how well the current path supports answering the question. Choose **one** of the following options **based on the definitions**:

                - "continue": Use this if the current path is on the right track but more steps are needed to reach the final answer.
                - "respond": Use this if the information collected so far is sufficient to answer the question.
                - "needs refinement": Use this if the current path is not helpful or has reached a dead end and should be restarted.

                Only respond with one of these exact words: "continue", "respond", or "needs refinement"."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip().lower()


def redefine_CoT_steps(exploration_plan, query):
    old_steps = "\n".join(exploration_plan)

    prompt = f"""You are assisting in knowledge graph-based question answering.

The previous exploration plan did not lead to the correct answer.

Question: "{query}"

Here was the previous plan:
{old_steps}

Please generate a new improved plan with 3–5 steps. The steps should focus on:
- Identifying key entities or nodes
- Exploring relevant relationships
- Filtering to find the right target

Return the new plan as a JSON object in the format:
{{ "steps": ["1. ...", "2. ...", "3. ..."] }}
Only return valid JSON.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=1,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    new_plan = json.loads(response.choices[0].message.content)
    return new_plan["steps"]
  

def generate_answer(path_traveled, query):
    steps_str = ""
    for step in path_traveled:
        step_text = step["step"]
        node_str = ", ".join(step.get("nodes", []))
        rel_str = ", ".join(step.get("rels", [])) if "rels" in step else ""
        if rel_str:
            steps_str += f"{step_text} -> [{rel_str}] -> {node_str}\n"
        else:
            steps_str += f"{step_text} -> {node_str}\n"

    prompt = f"""You are an expert assistant answering questions using information extracted from a knowledge graph.

Question: "{query}"

Based only on the traversal path below, generate a concise and factual answer. Only use the given information and do not add any other knowledge.

Path traveled:
{steps_str}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


def KG_RAG(QA_df, output_name):
    print("Start KG_RAG()")
    RAG_df = pd.DataFrame()    

    for i in range(QA_df.shape[0]):
    #for i in range(5):
        query = QA_df['Question'].iloc[i]        
        exploration_plan = plan_exploration_steps(query)
        #print(exploration_plan)
        current_nodes = []
        path_traveled = []
        current_step = 0
        failed_tries = 0
        
        result_paths = []
        answered = False
        
        
        while current_step < len(exploration_plan) and failed_tries < 3 and not answered:
            
            step = exploration_plan[current_step]
            #print("\nStep ", step, "\n")
            
            if is_node_exploration(step):
                #print("### Node exploration ###")
                node_candidates = vectorDB_search(step, top_k=5)
                #print("Node candidates: ", node_candidates)
                
                selected_nodes = select_relevant_nodes_with_llm(step, node_candidates)
                #print("Selected nodes: ", selected_nodes)
                current_nodes = selected_nodes
                path_traveled.append({'step': step, 'nodes': selected_nodes})
                #path_traveled.append({'nodes': selected_nodes})
                #print("Path traveled: ", path_traveled)
                                
            if is_relationship_exploration(step):
                #print("### Relationship exploration ###")
                
                explorable_rels = cypher_query_explorable_relationships(current_nodes)
                #print("Explorable rels: ", explorable_rels)
                
                rel_candidates = vector_similarity_rank(explorable_rels, step)
                #print("Rel candidates: ", rel_candidates)
                
                selected_rels = select_relevant_rels_with_llm(step, rel_candidates)
                #print("Selected rels: ", selected_rels)
                
                #print("Target nodes: ", current_nodes)
                #path_traveled.append({'step': step, 'rels': selected_rels, 'nodes': current_nodes})
                paths = get_target_nodes(current_nodes, selected_rels)
                path_traveled.append({"step": step, "paths": paths})
                for path in paths:
                    triplet = (path["source"], path["relation"], path["target"])
                    result_paths.append(triplet)
                #print("Path traveled: ", path_traveled)
                #print("Result_paths: ", result_paths)
                
            eval_outcome = eval_state_with_llm(path_traveled, query)
            #print("EVALUATION: ", eval_outcome, "\n")
            if eval_outcome == "needs refinement":
                exploration_plan = redefine_CoT_steps(exploration_plan, query)
                current_step = 1
                path_traveled = []
                failed_tries += 1
            elif eval_outcome == "continue":
                current_step += 1
            elif eval_outcome == "respond":
                answer = generate_answer(path_traveled, query)
                answered = True
                break
                #print(answer)
        
        if(failed_tries >= 3):
            result_paths = []
            answer = "I am sorry, I could not find an answer to your question."
            answered = True
        elif not answered:
            answer = generate_answer(path_traveled, query)
        
        #print("Final answer: ", answer)
        new_row = {'query':query, 'exploration_plan':exploration_plan, 'paths':result_paths, 'answer':answer}
        RAG_df = RAG_df._append(new_row, ignore_index=True)
        print(f"\rProgress: {i+1}/{QA_df.shape[0]}", end="", flush=True)
    
    print("\n2 Save result file ", output_name, "...")
    file_path_name = f"../evaluation/results/{output_name}.csv"
    RAG_df.to_csv(file_path_name, index=False, encoding='utf-8')
    print("END KG_RAG()")
    
    
if __name__  == "__main__":
    URI = "bolt://localhost:7687"
    USERNAME = "neo4j"
    PASSWORD = "875421963"
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    test_connection()
    
    
    kg = pd.read_csv(r"../data/Economic_KG.csv")
    nodes = set(kg['n']).union(set(kg['m']))
    nodes = list(nodes)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(nodes, convert_to_tensor=True)
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings.detach().cpu().numpy())
    index_to_node = {i: node for i, node in enumerate(nodes)}
    
    
    QA_descriptive = pd.read_csv(r"../data/QA_descriptive.csv")
    QA_singleEntity = pd.read_csv(r"../data/QA_singleEntity.csv")
    QA_yesNo = pd.read_csv(r"../data/QA_yesNo.csv")
    QA_NULL = pd.read_csv(r"../data/QA_NULL.csv")
    
    KG_RAG(QA_df=QA_descriptive, output_name='KG_RAG_descriptive')
    KG_RAG(QA_df=QA_singleEntity, output_name='KG_RAG_singleEntity')
    KG_RAG(QA_df=QA_yesNo, output_name='KG_RAG_yesNo')
    KG_RAG(QA_df=QA_NULL, output_name='KG_RAG_NULL')
    
    ''' 
    QA_converging = pd.read_csv(r"../data/QA_Converging.csv")
    QA_divergent = pd.read_csv(r"../data/QA_Divergent.csv")
    QA_linear = pd.read_csv(r"../data/QA_Linear.csv")
    
    
    KG_RAG(QA_df=QA_converging, output_name='KG_RAG_Converging')
    KG_RAG(QA_df=QA_divergent, output_name='KG_RAG_Divergent')
    KG_RAG(QA_df=QA_linear, output_name='KG_RAG_Linear')
    '''