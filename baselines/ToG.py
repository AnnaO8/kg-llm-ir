from neo4j import GraphDatabase
import os
import argparse
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from keybert import KeyBERT
import time
import re
import random


load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    
### Arguments
    
#the max length of LLMs output
max_length = 256
    
# the temperature in exploration stage
temperature_exploration = 0.4
    
# the temperature in reasoning stage
temperature_reasoning = 0
    
# choose the search width of ToG
width = 3
    
# choose the search depth of ToG
depth = 3
    
# whether removing unnecessary relations
remove_unnecessary_rel = True
    
# base LLM model
LLM_type = "gpt-4o-mini"
    
# Number of entities retained during entities search
num_retain_entity = 5
    
# prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert
prune_tools = "llm"


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


cot_prompt = """Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is {Washington, D.C.}.

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is {Bharoto Bhagyo Bidhata}.

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: First, the artist nominated for an award for You Drive Me Crazy is Britney Spears. The answer is {Jason Allen Alexander}.

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is {Peter Paul Rubens}.

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia. The answer is {Georgia}.

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is {Heroin}."""


extract_relation_prompt = """Please retrieve %s relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of %s relations is 1).
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Topic Entity: Brahui Language
Relations: language.human_language.main_country; language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region
A: 1. {language.human_language.main_country (Score: 0.4))}: This relation is highly relevant as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980.
2. {language.human_language.countries_spoken_in (Score: 0.3)}: This relation is also relevant as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president.
3. {base.rosetta.languoid.parent (Score: 0.2)}: This relation is less relevant but still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question.

Q: """


score_entity_candidates_prompt = """Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1).
Q: The movie featured Miley Cyrus and was produced by Tobin Armbrust?
Relation: film.producer.film
Entites: The Resident; So Undercover; Let Me In; Begin Again; The Quiet Ones; A Walk Among the Tombstones
Score: 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
The movie that matches the given criteria is "So Undercover" with Miley Cyrus and produced by Tobin Armbrust. Therefore, the score for "So Undercover" would be 1, and the scores for all other entities would be 0.

Q: {}
Relation: {}
Entites: """


answer_prompt = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question with these triplets and your knowledge.
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson
A: Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said "Taste cannot be controlled by law," which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead.

Q: The artist nominated for The Long Winter lived where?
Knowledge Triplets: The Long Winter, book.written_work.author, Laura Ingalls Wilder
Laura Ingalls Wilder, people.person.places_lived, Unknown-Entity
Unknown-Entity, people.place_lived.location, De Smet
A: Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet. Therefore, the answer to the question is {De Smet}.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
Steve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens
Steve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group
A: Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.

Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity
Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
Kenya, location.country.currency_used, Kenyan shilling
A: Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer to the question is {Kenyan shilling}.

Q: The country with the National Anthem of Bolivia borders which nations?
Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
Bolivia, location.country.national_anthem, UnName_Entity
A: Based on the given knowledge triplets, we can infer that the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia itself. However, the given knowledge triplets do not provide information about which nations border Bolivia. To answer this question, we need additional knowledge about the geography of Bolivia and its neighboring countries.

Q: {}
"""

prompt_evaluate="""Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge (Yes or No).
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson
A: {No}. Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said "Taste cannot be controlled by law," which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead.

Q: The artist nominated for The Long Winter lived where?
Knowledge Triplets: The Long Winter, book.written_work.author, Laura Ingalls Wilder
Laura Ingalls Wilder, people.person.places_lived, Unknown-Entity
Unknown-Entity, people.place_lived.location, De Smet
A: {Yes}. Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet. Therefore, the answer to the question is {De Smet}.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
Steve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens
Steve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group
A: {No}. Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.

Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity
Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
Kenya, location.country.currency_used, Kenyan shilling
A: {Yes}. Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer to the question is {Kenyan shilling}.

Q: The country with the National Anthem of Bolivia borders which nations?
Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
Bolivia, location.country.national_anthem, UnName_Entity
A: {No}. Based on the given knowledge triplets, we can infer that the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia itself. However, the given knowledge triplets do not provide information about which nations border Bolivia. To answer this question, we need additional knowledge about the geography of Bolivia and its neighboring countries.

"""


def run_llm(prompt, temperature, max_tokens, model="gpt-4o-mini"):
    '''
    if "llama" in engine.lower():
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"  # your local llama server port
        engine = openai.Model.list()["data"][0]["id"]
    else:
        openai.api_key = opeani_api_keys
    '''
    
    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},
                {"role":"user","content":prompt}]
    
    retries = 0
    max_retries = 1
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI error (attempt {retries+1}/{max_retries}): {e}")
            retries += 1
            time.sleep(2)
            return None
    
    '''
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI error: {e}. Retrying...")
            time.sleep(2)
    '''
    


def extract_topic_entities(query):
    keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1,3), stop_words='english')
    return [kw[0] for kw in keywords]

def generate_without_explored_paths(question):
    prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    response = run_llm(prompt, temperature_reasoning, max_length)
    return response


def construct_relation_prune_prompt(question, entity, total_relations):
    return extract_relation_prompt % (width, width) + question + '\nTopic Entity: ' + entity + '\nRelations: '+ '; '.join(total_relations) + "\nA: "


def clean_relations(string, entity, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations


def relation_search_prune(entity, pre_relations, pre_head, question):
    #sparql_relations_extract_head = sparql_head_relations % (entity_id)
    #head_relations = execurte_sparql(sparql_relations_extract_head)
    #head_relations = replace_relation_prefix(head_relations)
    with driver.session() as session:
        # Head: outgoing relations (entity)-[r]->()
        head_query = """
            MATCH (:Entity {name: $name})-[r]->()
            RETURN DISTINCT r.type AS relation
            """
        head_relations = session.run(head_query, name=entity)
        head_relations = [record["relation"] for record in head_relations]
        
        
        #sparql_relations_extract_tail= sparql_tail_relations % (entity_id)
        #tail_relations = execurte_sparql(sparql_relations_extract_tail)
        #tail_relations = replace_relation_prefix(tail_relations)
        
        # Tail: incoming relations ()-[r]->(entity)
        tail_query = """
            MATCH ()-[r]->(:Entity {name: $name})
            RETURN DISTINCT r.type AS relation
            """
        tail_relations = session.run(tail_query, name=entity)
        tail_relations = [record["relation"] for record in tail_relations]

    ### there are no unnecessary rels in Economic_KG -> skip this step 'remove_unnecessary_rel'
    #if remove_unnecessary_rel:
    #    head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
    #    tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    
    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations+tail_relations
    total_relations.sort()  # make sure the order in prompt is always equal
    
    if prune_tools == "llm":
        prompt = construct_relation_prune_prompt(question, entity, total_relations)

        result = run_llm(prompt, temperature_exploration, max_length, LLM_type)
        flag, retrieve_relations_with_scores = clean_relations(result, entity, head_relations) 

    #elif prune_tools == "bm25":
    #    topn_relations, topn_scores = compute_bm25_similarity(question, total_relations, width)
    #    flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity, head_relations) 
    #else:
    #    model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
    #    topn_relations, topn_scores = retrieve_top_docs(question, total_relations, model, width)
    #    flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations) 

    if flag:
        return retrieve_relations_with_scores
    else:
        return [] # format error or too small max_length



def entity_search(entity, relation, head=True):
    with driver.session() as session:
        if head:
            query = """
            MATCH (n {name: $name})-[r]->(m)
            WHERE r.type = $rel
            RETURN DISTINCT m.name AS candidate
            """
        else:
            query = """
            MATCH (n)-[r]->(m {name: $name})
            WHERE r.type = $rel
            RETURN DISTINCT n.name AS candidate
            """
        results = session.run(query, name=entity, rel=relation)
        return [record["candidate"] for record in results]


def construct_entity_score_prompt(question, relation, entity_candidates):
    return score_entity_candidates_prompt.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '

# Reihenfolge könnte durch LLM durcheinander gebracht werden, dann sind die Scores in anderer Reihenfolge als die Nodes, 
# aber es wird nicht bemerkt, da nodes und scores getrennt betrachtet werden
# dennoch keine Anpassung hier vorgenommen, da original ToG Code genutzt wird
def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        #print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)


def entity_score(question, entity_candidates, score, relation):
    if len(entity_candidates) == 1:
        return [score], entity_candidates
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates
    
    entity_candidates = sorted(entity_candidates)

    if prune_tools == "llm":
        prompt = construct_entity_score_prompt(question, relation, entity_candidates)

        result = run_llm(prompt, temperature_exploration, max_length)
        return [float(x) * score for x in clean_scores(result, entity_candidates)], entity_candidates

    #elif prune_tools == "bm25":
    #    topn_entities, topn_scores = compute_bm25_similarity(question, entity_candidates, width)
    #else:
    #    model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
    #    topn_entities, topn_scores = retrieve_top_docs(question, entity_candidates, model, width)
        
    #if if_all_zero(topn_scores):
    #    topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    #return [float(x) * score for x in topn_scores], topn_entities, entity_candidates_id


def update_history(entity_candidates, entity, scores, total_candidates, total_scores, total_relations, total_entities, total_topic_entities, total_head):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
    candidates_relation = [entity['relation']] * len(entity_candidates)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities.extend(entity_candidates)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_scores, total_relations, total_entities, total_topic_entities, total_head


def generate_answer(question, cluster_chain_of_entities): 
    prompt = answer_prompt + question + '\n'
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    result = run_llm(prompt, temperature_reasoning, max_length)
    return result


def half_stop(question, cluster_chain_of_entities, depth, RAG_df):
    #print("No new knowledge added during search depth %d, stop searching." % depth)
    answer = generate_answer(question, cluster_chain_of_entities)
    new_row = {'query': question, 'paths': cluster_chain_of_entities, 'answer': answer}
    RAG_df = RAG_df._append(new_row, ignore_index=True)
    return RAG_df
    #save_2_jsonl(question, answer, cluster_chain_of_entities, file_name=args.dataset)



def entity_prune(total_entities, total_relations, total_candidates, total_topic_entities, total_head, total_scores):
    zipped = list(zip(total_entities, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped]

    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:width], sorted_relations[:width], sorted_candidates[:width], sorted_topic_entities[:width], sorted_head[:width], sorted_scores[:width]
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
    if len(filtered_list) ==0:
        return False, [], [], [], []
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))

    tops = list(tops)
    cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    return True, cluster_chain_of_entities, entities_id, relations, heads


def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""


def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False


def reasoning(question, cluster_chain_of_entities):
    prompt = prompt_evaluate + question
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '

    response = run_llm(prompt, temperature_reasoning, max_length)
    
    result = extract_answer(response)
    if if_true(result):
        return True, response
    else:
        return False, response


def if_finish_list(lst):
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst


def ToG(QA_df):
    print("Start ToG()")
    RAG_df = pd.DataFrame()    

    for i in range(QA_df.shape[0]):
    #for i in range(10):
        print(f"\rStep: {i+1}/{QA_df.shape[0]}", end="", flush=True)
        query = QA_df['Question'].iloc[i]
        topic_entity_list = extract_topic_entities(query)
        topic_entity = {entity: entity for entity in topic_entity_list}
        result_paths = []
        
        cluster_chain_of_entities = []
        if len(topic_entity) == 0:
            answer = generate_without_explored_paths(query)
            new_row = {'query':query, 'paths':result_paths, 'answer': answer}
            RAG_df = RAG_df._append(new_row, ignore_index=True)
            continue
        pre_relations = []
        pre_heads= [-1] * len(topic_entity)
        flag_printed = False
        
        for d in range(1, depth+1):
            current_entity_relations_list = []
            i=0
            for entity in topic_entity:
                if entity!="[FINISH_ID]":
                    #retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations, pre_heads[i], query)  # best entity triplet, entitiy_id
                    retrieve_relations_with_scores = relation_search_prune(entity, pre_relations, pre_heads[i], query)  # best entity triplet, entitiy_id
                    current_entity_relations_list.extend(retrieve_relations_with_scores)
                i+=1
            total_candidates = []
            total_scores = []
            total_relations = []
            total_entities = []
            total_topic_entities = []
            total_head = []
            
            
            # Example for current_entity_relation_list
            # {
            #   "entity": "Earthquake",
            #   "relation": "has_location",
            #   "score": 0.91,
            #   "head": True
            # }
            
            for entity in current_entity_relations_list:
                if entity['head']:
                    entity_candidates = entity_search(entity['entity'], entity['relation'], True)
                else:
                    entity_candidates = entity_search(entity['entity'], entity['relation'], False)
                    
                    
                if prune_tools == "llm":
                    if len(entity_candidates) >=20:
                        entity_candidates = random.sample(entity_candidates, num_retain_entity)
                
                        
                if len(entity_candidates) ==0:
                    continue
                scores, entity_candidates = entity_score(query, entity_candidates, entity['score'], entity['relation'])
                
                total_candidates, total_scores, total_relations, total_entities, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, total_candidates, total_scores, total_relations, total_entities, total_topic_entities, total_head)



            if len(total_candidates) ==0:
                RAG_df = half_stop(query, cluster_chain_of_entities, d, RAG_df)
                flag_printed = True
                break
            
            flag, chain_of_entities, entities, pre_relations, pre_heads = entity_prune(total_entities, total_relations, total_candidates, total_topic_entities, total_head, total_scores)
            cluster_chain_of_entities.append(chain_of_entities)
            
            if flag:
                stop, results = reasoning(query, cluster_chain_of_entities)
                if stop:
                    #print("ToG stoped at depth %d." % d)
                    new_row = {'query': query, 'paths': cluster_chain_of_entities, 'answer': results}
                    RAG_df = RAG_df._append(new_row, ignore_index=True)
                    flag_printed = True
                    break
                else:
                    #print("depth %d still not find the answer." % d)
                    flag_finish, entities = if_finish_list(entities)
                    if flag_finish:
                        RAG_df = half_stop(query, cluster_chain_of_entities, d, RAG_df)
                        flag_printed = True
                    else:
                        topic_entity = {e: e for e in entities}
                        continue
            else:
                RAG_df = half_stop(query, cluster_chain_of_entities, d, RAG_df)
                flag_printed = True
                
        if not flag_printed:
            results = generate_without_explored_paths(query)
            new_row = {'query': query, 'paths': [], 'answer': results}
            RAG_df = RAG_df._append(new_row, ignore_index=True)
        
        
        #print(f"\rProgress: {i+1}/{QA_df.shape[0]}", end="", flush=True)
        
    print("\n2 Save file ToG_df_xxx.csv...")
    
    #RAG_df.to_csv('output/Conv/ToG_df_Converging_FAST.csv', index=False, encoding='utf-8')
    #RAG_df.to_csv('output/Div/ToG_df_Divergent_FAST.csv', index=False, encoding='utf-8')
    RAG_df.to_csv('output/Lin/ToG_df_Linear_FAST.csv', index=False, encoding='utf-8')
    print("END ToG()")



if __name__ == '__main__':
    URI = "bolt://localhost:7687"
    USERNAME = "neo4j"
    PASSWORD = "875421963"
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    test_connection()
    
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    
    ### Arguments
    
    #the max length of LLMs output
    max_length = 256
    
    # the temperature in exploration stage
    temperature_exploration = 0.4
    
    # the temperature in reasoning stage
    temperature_reasoning = 0
    
    # choose the search width of ToG
    width = 3
    
    # choose the search depth of ToG
    depth = 3
    
    # whether removing unnecessary relations
    remove_unnecessary_rel = True
    
    # base LLM model
    LLM_type = "gpt-4o-mini"
    
    # Number of entities retained during entities search
    num_retain_entity = 5
    
    # prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert
    prune_tools = "llm"
    

    
    #kg = pd.read_csv(r"../data/Economic_KG.csv")
    #nodes = set(kg['n']).union(set(kg['m']))
    #nodes = list(nodes)
    
    QA_converging = pd.read_csv(r"../data/QA_Converging.csv")
    #QA_divergent = pd.read_csv(r"../data/QA_Divergent.csv")
    #QA_linear = pd.read_csv(r"../data/QA_Linear.csv")
    
    ToG(QA_converging)
    
    #datas, question_string = prepare_dataset(args.dataset)
    #print("Start Running ToG on %s dataset." % args.dataset)
    
    
    
    
    
    
    
    
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="webqsp", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int,
                        default=3, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    args = parser.parse_args()
    '''
