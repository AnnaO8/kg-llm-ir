import re
import numpy as np
import pandas as pd
import ast


def compute_metrics(ground_truth, results, k=10):
    all_AP = []
    all_RR = []
    all_Hit = []

    for i in range(len(ground_truth)):
        gt_set = sorted(ground_truth[i])
        retrieved_set = list(sorted(results[i]))

        # Compute Average Precision (AP) for MAP
        relevant_count = 0
        AP_sum = 0
        for j, tripel in enumerate(retrieved_set):
            if tripel in gt_set:
                relevant_count += 1
                AP_sum += relevant_count / (j + 1) 
        
        AP = AP_sum / len(gt_set) if gt_set else 0.0
        all_AP.append(AP)

        # Compute Reciprocal Rank (RR) for MRR
        RR = 0.0
        for j, tripel in enumerate(retrieved_set): 
            if tripel in gt_set:
                RR = 1 / (j + 1) 
                break
        all_RR.append(RR)

        # Compute Hit@k (1 if at least one correct answer in top-k, else 0)
        hit_k = 1 if any(tripel in gt_set for tripel in retrieved_set[:k]) else 0
        all_Hit.append(hit_k)

    # Compute final scores
    MRR_score = np.mean(all_RR) if all_RR else 0.0
    MAP_score = np.mean(all_AP) if all_AP else 0.0
    Hit_at_k = np.mean(all_Hit) if all_Hit else 0.0

    
    return MRR_score, MAP_score, Hit_at_k



def extract_triples_from_subgraph(subgraph_str):
    subgraph_str = subgraph_str.strip().replace('""', '"')

    # Knoten: Extrahiere alles zwischen `(:` und dem ersten `{` oder Leerzeichen
    # Relationen: Extrahiere alles zwischen `[:` und `]`
    node_pattern = r"\(:\s*([\w\s\-\d]+)"
    relation_pattern = r"\[:\s*([\w\s\-\d]+)\s*\]"

    nodes = [match.strip().replace("_", " ") for match in re.findall(node_pattern, subgraph_str)]
    relations = [match.strip().replace("_", " ") for match in re.findall(relation_pattern, subgraph_str)]

    # Debugging
    if len(nodes) != len(relations) + 1:
        print("Error: Nodes and relations do not align correctly!")
        print("Subgraph String:", repr(subgraph_str))
        print("Extracted Nodes:", nodes)
        print("Extracted Relations:", relations)
        return []

    triples = [(nodes[i], relations[i], nodes[i + 1]) for i in range(len(relations))]

    return triples


def extract_triples_from_simple_path(path_str):
    # Splitte den Pfad an den Pfeilen
    parts = re.split(r'\s*-\[|\]->\s*', path_str.strip())
    
    nodes = parts[::2]   # Jedes zweite Element: Knoten
    relations = parts[1::2]  # Jedes zweite Element ab Index 1: Relationen
    
    # Sicherheit: Knotenanzahl muss Relationanzahl + 1 sein
    if len(nodes) != len(relations) + 1:
        print("Fehlerhafte Struktur im Pfad!")
        print("Nodes:", nodes)
        print("Relations:", relations)
        return []
    
    triples = [(nodes[i].strip(), relations[i].strip(), nodes[i + 1].strip()) for i in range(len(relations))]
    return triples




def retrieval_eval_NaiveRAG(method, QA_groundTruth_df, RAG_results_df, print_results=True):
    if QA_groundTruth_df.shape[0] != RAG_results_df.shape[0]:
        raise ValueError("Mismatched number of QA pairs and RAG results.")
    
    ground_truth = [[] for _ in range(QA_groundTruth_df.shape[0])]
    results = [[] for _ in range(RAG_results_df.shape[0])]
    
    if method in ['converging', 'divergent']:
        for i in range(QA_groundTruth_df.shape[0]):
            subgraph1_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Subgraph1'].iloc[i])
            subgraph2_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Subgraph2'].iloc[i])
            seen = set()
            combined = []
            for triplet in subgraph1_triples_list + subgraph2_triples_list:
                triplet_tuple = tuple(triplet)
                if triplet_tuple not in seen:
                    seen.add(triplet_tuple)
                    combined.append(triplet)
            ground_truth[i] = combined
                            
    elif method == 'linear':
        for i in range(QA_groundTruth_df.shape[0]):
            subgraph_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Path'].iloc[i])
            seen = set()
            ordered_triples = []
            for triplet in subgraph_triples_list:
                triplet_tuple = tuple(triplet)
                if triplet_tuple not in seen:
                    seen.add(triplet_tuple)
                    ordered_triples.append(triplet_tuple)
            ground_truth[i] = ordered_triples
            
    elif method == 'QA_types':
        for i in range(QA_groundTruth_df.shape[0]):
            path = QA_groundTruth_df['path'].iloc[i]
            triples_list = extract_triples_from_simple_path(path)
            #print("Path: ", path, "\nTriples-List: ", triples_list)
            seen = set()
            ordered_triples = []
            for triplet in triples_list:
                triplet_tuple = tuple(triplet)
                if triplet_tuple not in seen:
                    seen.add(triplet_tuple)
                    ordered_triples.append(triplet_tuple)
            ground_truth[i] = ordered_triples
            #print(ground_truth[i])
           
            
    
    else:
        raise ValueError("Invalid method. Choose from 'converging', 'divergent', or 'linear'.")
    
    
    
    for i in range(RAG_results_df.shape[0]):
        retrieved = RAG_results_df['Retrieved Document'].iloc[i]
        retrieved_list = ast.literal_eval(retrieved)
        cleaned_triples_list = [
            tuple(part.replace("_", " ") for part in triple)
            for triple in retrieved_list
        ]
        seen = set()
        ordered_results = []
        for tripel in cleaned_triples_list:
            
            #print("Tripel: ", tripel)
            if tripel not in seen:
                seen.add(tripel)
                ordered_results.append(tripel)

        results[i] = ordered_results
    
    MRR_score, MAP_score, Hit_at_k = compute_metrics(ground_truth, results)
    if print_results == True:
        print("Naive RAG EVALUATION - ", method)
        print("MRR:", round(MRR_score, 4),
            "MAP:", round(MAP_score, 4),
            f"Hit@10:", round(Hit_at_k, 4))
    
        print("\n")
    ###
    # resulted sets: ground_truth, results 

    #print("Ground Truth: ", ground_truth[20])
    #print("Results: ", results[20])
    return ground_truth, results, MRR_score, MAP_score, Hit_at_k   


def retrieval_eval_NaiveRAG_Subquery(method, QA_groundTruth_df, RAG_results_df, print_results=True):
    if QA_groundTruth_df.shape[0] != RAG_results_df.shape[0]:
        raise ValueError("Mismatched number of QA pairs and RAG results.")
    
    ground_truth = [[] for _ in range(QA_groundTruth_df.shape[0])]
    results = [[] for _ in range(RAG_results_df.shape[0])]
    
    if method in ['converging', 'divergent']:
        for i in range(QA_groundTruth_df.shape[0]):
            subgraph1_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Subgraph1'].iloc[i])
            subgraph2_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Subgraph2'].iloc[i])
            seen = set()
            combined = []
            for triplet in subgraph1_triples_list + subgraph2_triples_list:
                triplet_tuple = tuple(triplet)
                if triplet_tuple not in seen:
                    seen.add(triplet_tuple)
                    combined.append(triplet)
            ground_truth[i] = combined
                            
    elif method == 'linear':
        for i in range(QA_groundTruth_df.shape[0]):
            subgraph_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Path'].iloc[i])
            seen = set()
            ordered_triples = []
            for triplet in subgraph_triples_list:
                triplet_tuple = tuple(triplet)
                if triplet_tuple not in seen:
                    seen.add(triplet_tuple)
                    ordered_triples.append(triplet_tuple)
            ground_truth[i] = ordered_triples
    
    else:
        raise ValueError("Invalid method. Choose from 'converging', 'divergent', or 'linear'.")
    
    
    
    for i in range(RAG_results_df.shape[0]):
        retrieved = RAG_results_df['Retrieved Document'].iloc[i]
        retrieved_list = ast.literal_eval(retrieved)
        cleaned_triples_list = [
            tuple(part.replace("_", " ") for part in triple)
            for triple in retrieved_list
        ]
        seen = set()
        ordered_results = []
        for tripel in cleaned_triples_list:
            
            #print("Tripel: ", tripel)
            if tripel not in seen:
                seen.add(tripel)
                ordered_results.append(tripel)

        results[i] = ordered_results
    
    MRR_score, MAP_score, Hit_at_k = compute_metrics(ground_truth, results)
    if print_results == True:
        print("Naive RAG + Subquery EVALUATION - ", method)
        print("MRR:", round(MRR_score, 4),
            "MAP:", round(MAP_score, 4),
            f"Hit@10:", round(Hit_at_k, 4))
        
        print("\n")
    ###
    # resulted sets: ground_truth, results 

    #print("Ground Truth: ", ground_truth[20])
    #print("Results: ", results[20])
    return ground_truth, results, MRR_score, MAP_score, Hit_at_k


def retrieval_eval_HybridRAG(method, QA_groundTruth_df, RAG_results_df, print_results=True):
    if QA_groundTruth_df.shape[0] != RAG_results_df.shape[0]:
        raise ValueError("Mismatched number of QA pairs and RAG results.")
    
    ground_truth = [[] for _ in range(QA_groundTruth_df.shape[0])]
    results = [[] for _ in range(RAG_results_df.shape[0])]
    
    if method in ['converging', 'divergent']:
        for i in range(QA_groundTruth_df.shape[0]):
            subgraph1_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Subgraph1'].iloc[i])
            subgraph2_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Subgraph2'].iloc[i])
            seen = set()
            combined = []
            for triplet in subgraph1_triples_list + subgraph2_triples_list:
                triplet_tuple = tuple(triplet)
                if triplet_tuple not in seen:
                    seen.add(triplet_tuple)
                    combined.append(triplet)
            ground_truth[i] = combined
                            
    elif method == 'linear':
        for i in range(QA_groundTruth_df.shape[0]):
            subgraph_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Path'].iloc[i])
            seen = set()
            ordered_triples = []
            for triplet in subgraph_triples_list:
                triplet_tuple = tuple(triplet)
                if triplet_tuple not in seen:
                    seen.add(triplet_tuple)
                    ordered_triples.append(triplet_tuple)
            ground_truth[i] = ordered_triples
    
    else:
        raise ValueError("Invalid method. Choose from 'converging', 'divergent', or 'linear'.")
    
   
    for i in range(RAG_results_df.shape[0]):
        retrieved = RAG_results_df['combined context'].iloc[i]
        retrieved_list = ast.literal_eval(retrieved)
        cleaned_triples_list = [
            tuple(part.replace("_", " ") for part in triple)
            for triple in retrieved_list
        ]
        seen = set()
        ordered_results = []
        for tripel in cleaned_triples_list:
            
            #print("Tripel: ", tripel)
            if tripel not in seen:
                seen.add(tripel)
                ordered_results.append(tripel)

        results[i] = ordered_results
    
    MRR_score, MAP_score, Hit_at_k = compute_metrics(ground_truth, results)
    if print_results == True:
        print("Hybrid RAG EVALUATION - ", method)
        print("MRR:", round(MRR_score, 4),
            "MAP:", round(MAP_score, 4),
            f"Hit@10:", round(Hit_at_k, 4))
        
        print("\n")
    ###
    # resulted sets: ground_truth, results 

    #print("Ground Truth: ", ground_truth[20])
    #print("Results: ", results[20])
    return ground_truth, results, MRR_score, MAP_score, Hit_at_k


def retrieval_eval_RerankRAG(method, QA_groundTruth_df, RAG_results_df, print_results=True):
    if QA_groundTruth_df.shape[0] != RAG_results_df.shape[0]:
        raise ValueError("Mismatched number of QA pairs and RAG results.")
    
    ground_truth = [[] for _ in range(QA_groundTruth_df.shape[0])]
    results = [[] for _ in range(RAG_results_df.shape[0])]
    
    if method in ['converging', 'divergent']:
        for i in range(QA_groundTruth_df.shape[0]):
            subgraph1_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Subgraph1'].iloc[i])
            subgraph2_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Subgraph2'].iloc[i])
            seen = set()
            combined = []
            for triplet in subgraph1_triples_list + subgraph2_triples_list:
                triplet_tuple = tuple(triplet)
                if triplet_tuple not in seen:
                    seen.add(triplet_tuple)
                    combined.append(triplet)
            ground_truth[i] = combined
                            
    elif method == 'linear':
        for i in range(QA_groundTruth_df.shape[0]):
            subgraph_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Path'].iloc[i])
            seen = set()
            ordered_triples = []
            for triplet in subgraph_triples_list:
                triplet_tuple = tuple(triplet)
                if triplet_tuple not in seen:
                    seen.add(triplet_tuple)
                    ordered_triples.append(triplet_tuple)
            ground_truth[i] = ordered_triples
    
    else:
        raise ValueError("Invalid method. Choose from 'converging', 'divergent', or 'linear'.")
    
   
    for i in range(RAG_results_df.shape[0]):
        retrieved = RAG_results_df['reranked triples'].iloc[i]
        retrieved_list = ast.literal_eval(retrieved)
        cleaned_triples_list = [
            tuple(part.replace("_", " ") for part in triple)
            for triple in retrieved_list
        ]
        seen = set()
        ordered_results = []
        for tripel in cleaned_triples_list:
            
            #print("Tripel: ", tripel)
            if tripel not in seen:
                seen.add(tripel)
                ordered_results.append(tripel)

        results[i] = ordered_results
    
    
    MRR_score, MAP_score, Hit_at_k = compute_metrics(ground_truth, results)
    if print_results == True:
        print("Rerank RAG EVALUATION - ", method)
        print("MRR:", round(MRR_score, 4),
            "MAP:", round(MAP_score, 4),
            f"Hit@10:", round(Hit_at_k, 4))
        
        print("\n")
    ###
    # resulted sets: ground_truth, results 

    #print("Ground Truth: ", ground_truth[20])
    #print("Results: ", results[20])
    return ground_truth, results, MRR_score, MAP_score, Hit_at_k  





def extract_triples_with_direction(paths):
    triples = set()
    
    for path in paths:
        nodes = list(path)
        for i in range(0, len(nodes) - 2, 2):
            source = nodes[i]
            relation = nodes[i + 1]
            target = nodes[i + 2]
            
            # Standardisiere die Richtung der Beziehung
            if "->" in relation:
                triples.add((source, relation.replace(" ->",""), target))
            elif "<-" in relation:
                triples.add((target, relation.replace(" <-", ""), source))
    
    return triples



def retrieval_eval_Naive_GraphRAG(method, QA_groundTruth_df, RAG_results_df, print_results = True):
    if QA_groundTruth_df.shape[0] != RAG_results_df.shape[0]:
        raise ValueError("Mismatched number of QA pairs and RAG results.")
    
    #ground_truth = [set() for _ in range(QA_groundTruth_df.shape[0])]
    #results = [set() for _ in range(RAG_results_df.shape[0])]
    ground_truth = [[] for _ in range(QA_groundTruth_df.shape[0])]
    results = [[] for _ in range(RAG_results_df.shape[0])]
    
    if method in ['converging', 'divergent']:
        for i in range(QA_groundTruth_df.shape[0]):
            subgraph1_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Subgraph1'].iloc[i])
            subgraph2_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Subgraph2'].iloc[i])
            seen = set()
            combined = []
            for triplet in subgraph1_triples_list + subgraph2_triples_list:
                triplet_tuple = tuple(triplet)
                if triplet_tuple not in seen:
                    seen.add(triplet_tuple)
                    combined.append(triplet)
            ground_truth[i] = combined
                            
    elif method == 'linear':
        for i in range(QA_groundTruth_df.shape[0]):
            subgraph_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Path'].iloc[i])
            seen = set()
            ordered_triples = []
            for triplet in subgraph_triples_list:
                triplet_tuple = tuple(triplet)
                if triplet_tuple not in seen:
                    seen.add(triplet_tuple)
                    ordered_triples.append(triplet_tuple)

            ground_truth[i] = ordered_triples
    
    else:
        raise ValueError("Invalid method. Choose from 'converging', 'divergent', or 'linear'.")
    
    for i in range(RAG_results_df.shape[0]):
        retrieved = RAG_results_df['paths'].iloc[i]
        retrieved_list = ast.literal_eval(retrieved)

        extracted_triples = extract_triples_with_direction(retrieved_list)
        
        seen = set()
        ordered_results = []
        for triplet in extracted_triples:
            triplet_tuple = tuple(triplet)
            if triplet_tuple not in seen:
                seen.add(triplet_tuple)
                ordered_results.append(triplet_tuple)
                
        results[i] = ordered_results
          

    MRR_score, MAP_score, Hit_at_k = compute_metrics(ground_truth, results)
    if print_results == True:
        print("Naive GraphRAG EVALUATION - ", method) 
        print("MRR:", round(MRR_score, 4),
            "MAP:", round(MAP_score, 4),
            f"Hit@10:", round(Hit_at_k, 4))
        
        print("\n")

    return ground_truth, results, MRR_score, MAP_score, Hit_at_k


def retrieval_eval_KG_RAG(method, QA_groundTruth_df, RAG_results_df, print_results=True):
    if QA_groundTruth_df.shape[0] != RAG_results_df.shape[0]:
        raise ValueError("Mismatched number of QA pairs and RAG results.")
    
    #ground_truth = [set() for _ in range(QA_groundTruth_df.shape[0])]
    #results = [set() for _ in range(RAG_results_df.shape[0])]
    ground_truth = [[] for _ in range(QA_groundTruth_df.shape[0])]
    results = [[] for _ in range(RAG_results_df.shape[0])]
    
    if method in ['converging', 'divergent']:
        for i in range(QA_groundTruth_df.shape[0]):
            subgraph1_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Subgraph1'].iloc[i])
            subgraph2_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Subgraph2'].iloc[i])
            seen = set()
            combined = []
            for triplet in subgraph1_triples_list + subgraph2_triples_list:
                triplet_tuple = tuple(triplet)
                if triplet_tuple not in seen:
                    seen.add(triplet_tuple)
                    combined.append(triplet)
            ground_truth[i] = combined
                            
    elif method == 'linear':
        for i in range(QA_groundTruth_df.shape[0]):
            #ground_truth[i] = set(extract_triples_from_subgraph(QA_groundTruth_df['Path'].iloc[i]))
            subgraph_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Path'].iloc[i])
            seen = set()
            ordered_triples = []
            for triplet in subgraph_triples_list:
                triplet_tuple = tuple(triplet)
                if triplet_tuple not in seen:
                    seen.add(triplet_tuple)
                    ordered_triples.append(triplet_tuple)

            ground_truth[i] = ordered_triples
    
    else:
        raise ValueError("Invalid method. Choose from 'converging', 'divergent', or 'linear'.")
    #print("GT: ",ground_truth[0])
    
    
    counter_empty = 0
    for i in range(RAG_results_df.shape[0]):
        retrieved = RAG_results_df['paths'].iloc[i]
        retrieved_list = ast.literal_eval(retrieved)
        #print("retrieved: ", retrieved)
        if retrieved_list == []:
            counter_empty += 1
        
        seen = set()
        ordered_results = []
        for triplet in retrieved_list:
            #print("Triplet: ", triplet)
            if triplet not in seen:
                seen.add(triplet)
                ordered_results.append(triplet)
                
        results[i] = ordered_results
        
    #print("Results: ", results[0]) 
    print("Empty answers: ", counter_empty)   

    MRR_score, MAP_score, Hit_at_k = compute_metrics(ground_truth, results) 
    if print_results == True:
        print("KG_RAG EVALUATION - ", method)
        print("MRR:", round(MRR_score, 4),
            "MAP:", round(MAP_score, 4),
            f"Hit@10:", round(Hit_at_k, 4))
        
        print("\n")

    return ground_truth, results, MRR_score, MAP_score, Hit_at_k



def retrieval_eval_GraphTRACE(method, QA_groundTruth_df, RAG_results_df, print_results=True):
    if QA_groundTruth_df.shape[0] != RAG_results_df.shape[0]:
        raise ValueError("Mismatched number of QA pairs and RAG results.")
    
    #ground_truth = [set() for _ in range(QA_groundTruth_df.shape[0])]
    #results = [set() for _ in range(RAG_results_df.shape[0])]
    ground_truth = [[] for _ in range(QA_groundTruth_df.shape[0])]
    results = [[] for _ in range(RAG_results_df.shape[0])]
    
    if method in ['converging', 'divergent']:
        for i in range(QA_groundTruth_df.shape[0]):
            subgraph1_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Subgraph1'].iloc[i])
            subgraph2_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Subgraph2'].iloc[i])
            seen = set()
            combined = []
            for triplet in subgraph1_triples_list + subgraph2_triples_list:
                triplet_tuple = tuple(triplet)
                if triplet_tuple not in seen:
                    seen.add(triplet_tuple)
                    combined.append(triplet)
            #ground_truth[i] = set(subgraph1_triples_list) | set(subgraph2_triples_list)
            ground_truth[i] = combined
                            
    elif method == 'linear':
        for i in range(QA_groundTruth_df.shape[0]):
            #ground_truth[i] = set(extract_triples_from_subgraph(QA_groundTruth_df['Path'].iloc[i]))
            subgraph_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Path'].iloc[i])
            seen = set()
            ordered_triples = []
            for triplet in subgraph_triples_list:
                triplet_tuple = tuple(triplet)
                if triplet_tuple not in seen:
                    seen.add(triplet_tuple)
                    ordered_triples.append(triplet_tuple)

            ground_truth[i] = ordered_triples
    
    else:
        raise ValueError("Invalid method. Choose from 'converging', 'divergent', or 'linear'.")
    #print(ground_truth[0])
    
    for i in range(RAG_results_df.shape[0]):
        retrieved = RAG_results_df['paths'].iloc[i]
        retrieved_list = ast.literal_eval(retrieved)

        extracted_triples = extract_triples_with_direction(retrieved_list)
        
        seen = set()
        ordered_results = []
        for triplet in extracted_triples:
            triplet_tuple = tuple(triplet)
            if triplet_tuple not in seen:
                seen.add(triplet_tuple)
                ordered_results.append(triplet_tuple)
                
        results[i] = ordered_results


    MRR_score, MAP_score, Hit_at_k = compute_metrics(ground_truth, results)
    if print_results == True:
        print("GraphTRACE EVALUATION - ", method)
        print("MRR:", round(MRR_score, 4),
            "MAP:", round(MAP_score, 4),
            f"Hit@10:", round(Hit_at_k, 4))
        
        print("\n") 

    return ground_truth, results, MRR_score, MAP_score, Hit_at_k



def retrieval_eval_GraphTRACE_v2(method, QA_groundTruth_df, RAG_results_df, print_results=True):
    if QA_groundTruth_df.shape[0] != RAG_results_df.shape[0]:
        raise ValueError("Mismatched number of QA pairs and RAG results.")
    
    #ground_truth = [set() for _ in range(QA_groundTruth_df.shape[0])]
    #results = [set() for _ in range(RAG_results_df.shape[0])]
    ground_truth = [[] for _ in range(QA_groundTruth_df.shape[0])]
    results = [[] for _ in range(RAG_results_df.shape[0])]
    
    if method in ['converging', 'divergent']:
        for i in range(QA_groundTruth_df.shape[0]):
            subgraph1_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Subgraph1'].iloc[i])
            subgraph2_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Subgraph2'].iloc[i])
            seen = set()
            combined = []
            for triplet in subgraph1_triples_list + subgraph2_triples_list:
                triplet_tuple = tuple(triplet)
                if triplet_tuple not in seen:
                    seen.add(triplet_tuple)
                    combined.append(triplet)
            ground_truth[i] = combined
                            
    elif method == 'linear':
        for i in range(QA_groundTruth_df.shape[0]):
            #ground_truth[i] = set(extract_triples_from_subgraph(QA_groundTruth_df['Path'].iloc[i]))
            subgraph_triples_list = extract_triples_from_subgraph(QA_groundTruth_df['Path'].iloc[i])
            seen = set()
            ordered_triples = []
            for triplet in subgraph_triples_list:
                triplet_tuple = tuple(triplet)
                if triplet_tuple not in seen:
                    seen.add(triplet_tuple)
                    ordered_triples.append(triplet_tuple)

            ground_truth[i] = ordered_triples
    
    else:
        raise ValueError("Invalid method. Choose from 'converging', 'divergent', or 'linear'.")
    #print(ground_truth[0])
    
    for i in range(RAG_results_df.shape[0]):
        retrieved = RAG_results_df['paths'].iloc[i]
        retrieved_list = ast.literal_eval(retrieved)

        extracted_triples = extract_triples_with_direction(retrieved_list)
        
        seen = set()
        ordered_results = []
        for triplet in extracted_triples:
            triplet_tuple = tuple(triplet)
            if triplet_tuple not in seen:
                seen.add(triplet_tuple)
                ordered_results.append(triplet_tuple)
                
        results[i] = ordered_results


    MRR_score, MAP_score, Hit_at_k = compute_metrics(ground_truth, results)
    if print_results == True:
        print("GraphTRACE_v2 EVALUATION - ", method)
        print("MRR:", round(MRR_score, 4),
            "MAP:", round(MAP_score, 4),
            f"Hit@10:", round(Hit_at_k, 4))
        
        print("\n") 

    return ground_truth, results, MRR_score, MAP_score, Hit_at_k


def overall_retrieval_eval_NaiveRAG():
    ground_truth_all = []
    results_all = []

    for method, qa_path, rag_path in [
        ('converging', "../data/QA_Converging.csv", "results/NaiveRAG_Converging.csv"),
        ('divergent', "../data/QA_Divergent.csv", "results/NaiveRAG_Divergent.csv"),
        ('linear', "../data/QA_Linear.csv", "results/NaiveRAG_Linear.csv")
    ]:
        QA_groundTruth_df = pd.read_csv(qa_path)
        RAG_results_df = pd.read_csv(rag_path)
    
        ground_truth, results, _, _, _ = retrieval_eval_NaiveRAG(method, QA_groundTruth_df, RAG_results_df)
    
        ground_truth_all.extend(ground_truth)
        results_all.extend(results)

    
    print("# OVERALL NaiveRAG #")
    MRR_score, MAP_score, Hit_at_k = compute_metrics(ground_truth_all, results_all)
    print("MRR:", round(MRR_score, 4),
          "MAP:", round(MAP_score, 4),
          f"Hit@10:", round(Hit_at_k, 4))
    
    print("\n")
    return ground_truth_all, results_all, MRR_score, MAP_score, Hit_at_k


def overall_retrieval_eval_NaiveRAG_Subquery():
    ground_truth_all = []
    results_all = []

    for method, qa_path, rag_path in [
        ('converging', "../data/QA_Converging.csv", "results/NaiveRAG_Subquery_Converging.csv"),
        ('divergent', "../data/QA_Divergent.csv", "results/NaiveRAG_Subquery_Divergent.csv"),
        ('linear', "../data/QA_Linear.csv", "results/NaiveRAG_Subquery_Linear.csv")
    ]:
        QA_groundTruth_df = pd.read_csv(qa_path)
        RAG_results_df = pd.read_csv(rag_path)
    
        ground_truth, results, _, _, _ = retrieval_eval_NaiveRAG_Subquery(method, QA_groundTruth_df, RAG_results_df)
    
        ground_truth_all.extend(ground_truth)
        results_all.extend(results)

    
    print("# OVERALL NaiveRAG Subquery #")
    MRR_score, MAP_score, Hit_at_k = compute_metrics(ground_truth_all, results_all)
    print("MRR:", round(MRR_score, 4),
          "MAP:", round(MAP_score, 4),
          f"Hit@10:", round(Hit_at_k, 4))
    
    print("\n")
    return ground_truth_all, results_all, MRR_score, MAP_score, Hit_at_k


def overall_retrieval_eval_HybridRAG():
    ground_truth_all = []
    results_all = []

    for method, qa_path, rag_path in [
        ('converging', "../data/QA_Converging.csv", "results/HybridRAG_Converging.csv"),
        ('divergent', "../data/QA_Divergent.csv", "results/HybridRAG_Divergent.csv"),
        ('linear', "../data/QA_Linear.csv", "results/HybridRAG_Linear.csv")
    ]:
        QA_groundTruth_df = pd.read_csv(qa_path)
        RAG_results_df = pd.read_csv(rag_path)
    
        ground_truth, results, _, _, _ = retrieval_eval_HybridRAG(method, QA_groundTruth_df, RAG_results_df)
    
        ground_truth_all.extend(ground_truth)
        results_all.extend(results)

    
    print("# OVERALL HybridRAG #")
    MRR_score, MAP_score, Hit_at_k = compute_metrics(ground_truth_all, results_all)
    print("MRR:", round(MRR_score, 4),
          "MAP:", round(MAP_score, 4),
          f"Hit@10:", round(Hit_at_k, 4))
    
    print("\n")
    return ground_truth_all, results_all, MRR_score, MAP_score, Hit_at_k


def overall_retrieval_eval_RerankRAG():
    ground_truth_all = []
    results_all = []

    for method, qa_path, rag_path in [
        ('converging', "../data/QA_Converging.csv", "results/RerankRAG_Converging.csv"),
        ('divergent', "../data/QA_Divergent.csv", "results/RerankRAG_Divergent.csv"),
        ('linear', "../data/QA_Linear.csv", "results/RerankRAG_Linear.csv")
    ]:
        QA_groundTruth_df = pd.read_csv(qa_path)
        RAG_results_df = pd.read_csv(rag_path)
    
        ground_truth, results, _, _, _ = retrieval_eval_RerankRAG(method, QA_groundTruth_df, RAG_results_df)
    
        ground_truth_all.extend(ground_truth)
        results_all.extend(results)

    
    print("# OVERALL RerankRAG #")
    MRR_score, MAP_score, Hit_at_k = compute_metrics(ground_truth_all, results_all)
    print("MRR:", round(MRR_score, 4),
          "MAP:", round(MAP_score, 4),
          f"Hit@10:", round(Hit_at_k, 4))
    
    print("\n")
    return ground_truth_all, results_all, MRR_score, MAP_score, Hit_at_k


def overall_retrieval_eval_Naive_GraphRAG():
    ground_truth_all = []
    results_all = []

    for method, qa_path, rag_path in [
        ('converging', "../data/QA_Converging.csv", "results/Naive_GraphRAG_Converging.csv"),
        ('divergent', "../data/QA_Divergent.csv", "results/Naive_GraphRAG_Divergent.csv"),
        ('linear', "../data/QA_Linear.csv", "results/Naive_GraphRAG_Linear.csv")
    ]:
        QA_groundTruth_df = pd.read_csv(qa_path)
        RAG_results_df = pd.read_csv(rag_path)
    
        ground_truth, results, _, _, _ = retrieval_eval_Naive_GraphRAG(method, QA_groundTruth_df, RAG_results_df)
    
        ground_truth_all.extend(ground_truth)
        results_all.extend(results)

    
    print("# OVERALL Naive GraphRAG #")
    MRR_score, MAP_score, Hit_at_k = compute_metrics(ground_truth_all, results_all)
    print("MRR:", round(MRR_score, 4),
          "MAP:", round(MAP_score, 4),
          f"Hit@10:", round(Hit_at_k, 4))
    
    print("\n")
    return ground_truth_all, results_all, MRR_score, MAP_score, Hit_at_k


def overall_retrieval_eval_KG_RAG():
    ground_truth_all = []
    results_all = []

    for method, qa_path, rag_path in [
        ('converging', "../data/QA_Converging.csv", "results/KG_RAG_Converging.csv"),
        ('divergent', "../data/QA_Divergent.csv", "results/KG_RAG_Divergent.csv"),
        ('linear', "../data/QA_Linear.csv", "results/KG_RAG_Linear.csv")
    ]:
        QA_groundTruth_df = pd.read_csv(qa_path)
        RAG_results_df = pd.read_csv(rag_path)
    
        ground_truth, results, _, _, _ = retrieval_eval_KG_RAG(method, QA_groundTruth_df, RAG_results_df)
    
        ground_truth_all.extend(ground_truth)
        results_all.extend(results)

    
    print("# OVERALL KG RAG #")
    MRR_score, MAP_score, Hit_at_k = compute_metrics(ground_truth_all, results_all)
    print("MRR:", round(MRR_score, 4),
          "MAP:", round(MAP_score, 4),
          f"Hit@10:", round(Hit_at_k, 4))
    
    print("\n")
    return ground_truth_all, results_all, MRR_score, MAP_score, Hit_at_k


def overall_retrieval_eval_GraphTRACE():
    ground_truth_all = []
    results_all = []

    for method, qa_path, rag_path in [
        ('converging', "../data/QA_Converging.csv", "results/GraphTRACE_Converging.csv"),
        ('divergent', "../data/QA_Divergent.csv", "results/GraphTRACE_Divergent.csv"),
        ('linear', "../data/QA_Linear.csv", "results/GraphTRACE_Linear.csv")
    ]:
        QA_groundTruth_df = pd.read_csv(qa_path)
        RAG_results_df = pd.read_csv(rag_path)
    
        ground_truth, results, _, _, _ = retrieval_eval_GraphTRACE(method, QA_groundTruth_df, RAG_results_df)
    
        ground_truth_all.extend(ground_truth)
        results_all.extend(results)

    
    print("# OVERALL GraphTRACE #")
    MRR_score, MAP_score, Hit_at_k = compute_metrics(ground_truth_all, results_all)
    print("MRR:", round(MRR_score, 4),
          "MAP:", round(MAP_score, 4),
          f"Hit@10:", round(Hit_at_k, 4))
    
    print("\n")
    return ground_truth_all, results_all, MRR_score, MAP_score, Hit_at_k




def retrieval_eval_by_hop_NaiveRAG(method, qa_path, rag_path):
    QA_groundTruth_df = pd.read_csv(qa_path)
    RAG_results_df = pd.read_csv(rag_path)

    hop_values = sorted(QA_groundTruth_df['hop_count'].unique())

    hop_metrics = []

    for hop in hop_values:
        # Filtere QA- und RAG-Ergebnisse nach hop_count
        QA_filtered = QA_groundTruth_df[QA_groundTruth_df['hop_count'] == hop].reset_index(drop=True)
        RAG_filtered = RAG_results_df[RAG_results_df['query'].isin(QA_filtered['Question'])].copy()
        print("# QA count for ", hop, "-hop: ", RAG_filtered.shape[0])

        #print(QA_filtered.head())
        #print(RAG_filtered.head())
        
        if QA_filtered.empty or RAG_filtered.empty:
            print("check empty")
            continue

        # Berechne Retrieval-Metriken für diese Hop-Stufe
        ground_truth, results, MRR, MAP, Hit_at_k = retrieval_eval_NaiveRAG(method, QA_filtered, RAG_filtered, print_results=False)

        print(f"Hop Count {hop}: MRR={MRR:.4f}, MAP={MAP:.4f}, Hit@10={Hit_at_k:.4f}")
        hop_metrics.append((hop, MRR, MAP, Hit_at_k))

    return hop_metrics


def retrieval_eval_by_hop_NaiveRAG_Subquery(method, qa_path, rag_path):
    QA_groundTruth_df = pd.read_csv(qa_path)
    RAG_results_df = pd.read_csv(rag_path)

    hop_values = sorted(QA_groundTruth_df['hop_count'].unique())

    hop_metrics = []

    for hop in hop_values:
        # Filtere QA- und RAG-Ergebnisse nach hop_count
        QA_filtered = QA_groundTruth_df[QA_groundTruth_df['hop_count'] == hop].reset_index(drop=True)
        RAG_filtered = RAG_results_df[RAG_results_df['query'].isin(QA_filtered['Question'])].copy()
        print("# QA count for ", hop, "-hop: ", RAG_filtered.shape[0])

        #print(QA_filtered.head())
        #print(RAG_filtered.head())
        
        if QA_filtered.empty or RAG_filtered.empty:
            print("check empty")
            continue

        # Berechne Retrieval-Metriken für diese Hop-Stufe
        ground_truth, results, MRR, MAP, Hit_at_k = retrieval_eval_NaiveRAG_Subquery(method, QA_filtered, RAG_filtered, print_results=False)

        print(f"Hop Count {hop}: MRR={MRR:.4f}, MAP={MAP:.4f}, Hit@10={Hit_at_k:.4f}")
        hop_metrics.append((hop, MRR, MAP, Hit_at_k))

    return hop_metrics


def retrieval_eval_by_hop_HybridRAG(method, qa_path, rag_path):
    QA_groundTruth_df = pd.read_csv(qa_path)
    RAG_results_df = pd.read_csv(rag_path)

    hop_values = sorted(QA_groundTruth_df['hop_count'].unique())

    hop_metrics = []

    for hop in hop_values:
        # Filtere QA- und RAG-Ergebnisse nach hop_count
        QA_filtered = QA_groundTruth_df[QA_groundTruth_df['hop_count'] == hop].reset_index(drop=True)
        RAG_filtered = RAG_results_df[RAG_results_df['query'].isin(QA_filtered['Question'])].copy()
        print("# QA count for ", hop, "-hop: ", RAG_filtered.shape[0])

        #print(QA_filtered.head())
        #print(RAG_filtered.head())
        
        if QA_filtered.empty or RAG_filtered.empty:
            print("check empty")
            continue

        # Berechne Retrieval-Metriken für diese Hop-Stufe
        ground_truth, results, MRR, MAP, Hit_at_k = retrieval_eval_HybridRAG(method, QA_filtered, RAG_filtered, print_results=False)

        print(f"Hop Count {hop}: MRR={MRR:.4f}, MAP={MAP:.4f}, Hit@10={Hit_at_k:.4f}")
        hop_metrics.append((hop, MRR, MAP, Hit_at_k))

    return hop_metrics


def retrieval_eval_by_hop_RerankRAG(method, qa_path, rag_path):
    QA_groundTruth_df = pd.read_csv(qa_path)
    RAG_results_df = pd.read_csv(rag_path)

    hop_values = sorted(QA_groundTruth_df['hop_count'].unique())

    hop_metrics = []

    for hop in hop_values:
        # Filtere QA- und RAG-Ergebnisse nach hop_count
        QA_filtered = QA_groundTruth_df[QA_groundTruth_df['hop_count'] == hop].reset_index(drop=True)
        RAG_filtered = RAG_results_df[RAG_results_df['query'].isin(QA_filtered['Question'])].copy()
        print("# QA count for ", hop, "-hop: ", RAG_filtered.shape[0])

        #print(QA_filtered.head())
        #print(RAG_filtered.head())
        
        if QA_filtered.empty or RAG_filtered.empty:
            print("check empty")
            continue

        # Berechne Retrieval-Metriken für diese Hop-Stufe
        ground_truth, results, MRR, MAP, Hit_at_k = retrieval_eval_RerankRAG(method, QA_filtered, RAG_filtered, print_results=False)

        print(f"Hop Count {hop}: MRR={MRR:.4f}, MAP={MAP:.4f}, Hit@10={Hit_at_k:.4f}")
        hop_metrics.append((hop, MRR, MAP, Hit_at_k))

    return hop_metrics



def retrieval_eval_by_hop_Naive_GraphRAG(method, qa_path, rag_path):
    QA_groundTruth_df = pd.read_csv(qa_path)
    RAG_results_df = pd.read_csv(rag_path)

    hop_values = sorted(QA_groundTruth_df['hop_count'].unique())

    hop_metrics = []

    for hop in hop_values:
        # Filtere QA- und RAG-Ergebnisse nach hop_count
        QA_filtered = QA_groundTruth_df[QA_groundTruth_df['hop_count'] == hop].reset_index(drop=True)
        RAG_filtered = RAG_results_df[RAG_results_df['query'].isin(QA_filtered['Question'])].copy()
        print("# QA count for ", hop, "-hop: ", RAG_filtered.shape[0])

        #print(QA_filtered.head())
        #print(RAG_filtered.head())
        
        if QA_filtered.empty or RAG_filtered.empty:
            print("check empty")
            continue

        # Berechne Retrieval-Metriken für diese Hop-Stufe
        ground_truth, results, MRR, MAP, Hit_at_k = retrieval_eval_Naive_GraphRAG(method, QA_filtered, RAG_filtered, print_results=False)

        print(f"Hop Count {hop}: MRR={MRR:.4f}, MAP={MAP:.4f}, Hit@10={Hit_at_k:.4f}")
        hop_metrics.append((hop, MRR, MAP, Hit_at_k))

    return hop_metrics



def retrieval_eval_by_hop_KG_RAG(method, qa_path, rag_path):
    QA_groundTruth_df = pd.read_csv(qa_path)
    RAG_results_df = pd.read_csv(rag_path)

    hop_values = sorted(QA_groundTruth_df['hop_count'].unique())

    hop_metrics = []

    for hop in hop_values:
        # Filtere QA- und RAG-Ergebnisse nach hop_count
        QA_filtered = QA_groundTruth_df[QA_groundTruth_df['hop_count'] == hop].reset_index(drop=True)
        RAG_filtered = RAG_results_df[RAG_results_df['query'].isin(QA_filtered['Question'])].copy()
        print("# QA count for ", hop, "-hop: ", RAG_filtered.shape[0])

        #print(QA_filtered.head())
        #print(RAG_filtered.head())
        
        if QA_filtered.empty or RAG_filtered.empty:
            print("check empty")
            continue

        # Berechne Retrieval-Metriken für diese Hop-Stufe
        ground_truth, results, MRR, MAP, Hit_at_k = retrieval_eval_KG_RAG(method, QA_filtered, RAG_filtered, print_results=False)

        print(f"Hop Count {hop}: MRR={MRR:.4f}, MAP={MAP:.4f}, Hit@10={Hit_at_k:.4f}")
        hop_metrics.append((hop, MRR, MAP, Hit_at_k))

    return hop_metrics

def retrieval_eval_by_hop_GraphTRACE(method, qa_path, rag_path):
    QA_groundTruth_df = pd.read_csv(qa_path)
    RAG_results_df = pd.read_csv(rag_path)

    hop_values = sorted(QA_groundTruth_df['hop_count'].unique())

    hop_metrics = []

    for hop in hop_values:
        # Filtere QA- und RAG-Ergebnisse nach hop_count
        QA_filtered = QA_groundTruth_df[QA_groundTruth_df['hop_count'] == hop].reset_index(drop=True)
        RAG_filtered = RAG_results_df[RAG_results_df['query'].isin(QA_filtered['Question'])].copy()
        print("# QA count for ", hop,"-hop: ", RAG_filtered.shape[0])

        #print(QA_filtered.head())
        #print(RAG_filtered.head())
        
        if QA_filtered.empty or RAG_filtered.empty:
            print("check empty")
            continue

        # Berechne Retrieval-Metriken für diese Hop-Stufe
        ground_truth, results, MRR, MAP, Hit_at_k = retrieval_eval_GraphTRACE(method, QA_filtered, RAG_filtered, print_results=False)

        print(f"Hop Count {hop}: MRR={MRR:.4f}, MAP={MAP:.4f}, Hit@10={Hit_at_k:.4f}")
        hop_metrics.append((hop, MRR, MAP, Hit_at_k))

    return hop_metrics



if __name__ == '__main__':
    #### Hop Count - Converging
    print("\n==== METRICS - HOP COUNT - CONVERGING ====")
    
    qa_path = "../data/QA_Converging.csv"
    rag_path = "results/NaiveRAG_Converging.csv"
    print("Naive RAG - Converging - Hop Count")
    retrieval_eval_by_hop_NaiveRAG('converging', qa_path, rag_path)
    
    rag_path = "results/NaiveRAG_Subquery_Converging.csv"
    print("\nNaive RAG Subquery - Converging - Hop Count")
    retrieval_eval_by_hop_NaiveRAG_Subquery('converging', qa_path, rag_path)
    
    rag_path = "results/HybridRAG_Converging.csv"
    print("\nHybrid RAG - Converging - Hop Count")
    retrieval_eval_by_hop_HybridRAG('converging', qa_path, rag_path)
    
    rag_path = "results/RerankRAG_Converging.csv"
    print("\nRerank RAG - Converging - Hop Count")
    retrieval_eval_by_hop_RerankRAG('converging', qa_path, rag_path)
    
    rag_path = "results/Naive_GraphRAG_Converging.csv"
    print("\nNaive GraphRAG - Converging - Hop Count")
    retrieval_eval_by_hop_Naive_GraphRAG('converging', qa_path, rag_path)
    
    rag_path = "results/KG_RAG_Converging.csv"
    print("\nKG_RAG - Converging - Hop Count")
    retrieval_eval_by_hop_KG_RAG('converging', qa_path, rag_path)
    
    rag_path = "results/GraphTRACE_Converging.csv"
    print("\nGraphTRACE - Converging - Hop Count")
    retrieval_eval_by_hop_GraphTRACE('converging', qa_path, rag_path)
    
    

    #### Hop Count - Divergent
    print("\n==== METRICS - HOP COUNT - Divergent ====")
    
    qa_path = "../data/QA_Divergent.csv"
    rag_path = "results/NaiveRAG_Divergent.csv"
    print("\nNaive RAG - Divergent - Hop Count")
    retrieval_eval_by_hop_NaiveRAG('divergent', qa_path, rag_path)
    
    rag_path = "results/NaiveRAG_Subquery_Divergent.csv"
    print("\nNaive RAG Subquery - Divergent - Hop Count")
    retrieval_eval_by_hop_NaiveRAG_Subquery('divergent', qa_path, rag_path)
    
    rag_path = "results/HybridRAG_Divergent.csv"
    print("\nHybrid RAG - Divergent - Hop Count")
    retrieval_eval_by_hop_HybridRAG('divergent', qa_path, rag_path)
    
    rag_path = "results/RerankRAG_Divergent.csv"
    print("\nRerank RAG - Divergent - Hop Count")
    retrieval_eval_by_hop_RerankRAG('divergent', qa_path, rag_path)
    
    rag_path = "results/Naive_GraphRAG_Divergent.csv"
    print("\nNaive GraphRAG - Divergent - Hop Count")
    retrieval_eval_by_hop_Naive_GraphRAG('divergent', qa_path, rag_path)
    
    rag_path = "results/KG_RAG_Divergent.csv"
    print("\nKG_RAG - Divergent - Hop Count")
    retrieval_eval_by_hop_KG_RAG('divergent', qa_path, rag_path)
    
    rag_path = "results/GraphTRACE_Divergent.csv"
    print("\nGraphTRACE - Divergent - Hop Count")
    retrieval_eval_by_hop_GraphTRACE('divergent', qa_path, rag_path)
    
    
    #### Hop Count - Linear
    print("\n==== METRICS - HOP COUNT - Linear ====")
    
    qa_path = "../data/QA_Linear.csv"
    rag_path = "results/NaiveRAG_Linear.csv"
    print("\nNaive RAG - Linear - Hop Count")
    retrieval_eval_by_hop_NaiveRAG('linear', qa_path, rag_path)
    
    rag_path = "results/NaiveRAG_Subquery_Linear.csv"
    print("\nNaive RAG Subquery - Linear - Hop Count")
    retrieval_eval_by_hop_NaiveRAG_Subquery('linear', qa_path, rag_path)
    
    rag_path = "results/HybridRAG_Linear.csv"
    print("\nHybrid RAG - Linear - Hop Count")
    retrieval_eval_by_hop_HybridRAG('linear', qa_path, rag_path)
    
    rag_path = "results/RerankRAG_Linear.csv"
    print("\nRerank RAG - Linear - Hop Count")
    retrieval_eval_by_hop_RerankRAG('linear', qa_path, rag_path)
    
    rag_path = "results/Naive_GraphRAG_Linear.csv"
    print("\nNaive GraphRAG - Linear - Hop Count")
    retrieval_eval_by_hop_Naive_GraphRAG('linear', qa_path, rag_path)
    
    rag_path = "results/KG_RAG_Linear.csv"
    print("\nKG_RAG - Linear - Hop Count")
    retrieval_eval_by_hop_KG_RAG('linear', qa_path, rag_path)
    
    rag_path = "results/GraphTRACE_Linear.csv"
    print("\nGraphTRACE - Linear - Hop Count")
    retrieval_eval_by_hop_GraphTRACE('linear', qa_path, rag_path)
    
    
    ############################################## 
    
    ### Overall Metrics with all methods
    print("\n==== OVERALL METRICS ====")
    overall_retrieval_eval_NaiveRAG()
    overall_retrieval_eval_NaiveRAG_Subquery()
    overall_retrieval_eval_HybridRAG()
    overall_retrieval_eval_RerankRAG()
    overall_retrieval_eval_Naive_GraphRAG()
    overall_retrieval_eval_KG_RAG()
    overall_retrieval_eval_GraphTRACE()
    print("==== END ====")

    
    ''' 
    ### Naive RAG - Converging
    QA_groundTruth_df = pd.read_csv("../data/QA_Converging.csv")
    RAG_results_df = pd.read_csv("results/NaiveRAG_Converging.csv")
    retrieval_eval_NaiveRAG('converging', QA_groundTruth_df, RAG_results_df)
    
    ### Naive RAG - Divergent
    QA_groundTruth_df = pd.read_csv("../data/QA_Divergent.csv")
    RAG_results_df = pd.read_csv("results/NaiveRAG_Divergent.csv")
    retrieval_eval_NaiveRAG('divergent', QA_groundTruth_df, RAG_results_df)
    
    ### Naive RAG - Linear
    QA_groundTruth_df = pd.read_csv("../data/QA_Linear.csv")
    RAG_results_df = pd.read_csv("results/NaiveRAG_Linear.csv")
    retrieval_eval_NaiveRAG('linear', QA_groundTruth_df, RAG_results_df)
    
    
    ##############################################    
    
    ### Naive RAG Subquery - Converging
    QA_groundTruth_df = pd.read_csv("../data/QA_Converging.csv")
    RAG_results_df = pd.read_csv("results/NaiveRAG_Subquery_Converging.csv")
    retrieval_eval_NaiveRAG_Subquery('converging', QA_groundTruth_df, RAG_results_df)
    
    ### Naive RAG Subquery - Divergent
    QA_groundTruth_df = pd.read_csv("../data/QA_Divergent.csv")
    RAG_results_df = pd.read_csv("results/NaiveRAG_Subquery_Divergent.csv")
    retrieval_eval_NaiveRAG_Subquery('divergent', QA_groundTruth_df, RAG_results_df)
    
    ### Naive RAG Subquery - Linear
    QA_groundTruth_df = pd.read_csv("../data/QA_Linear.csv")
    RAG_results_df = pd.read_csv("results/NaiveRAG_Subquery_Linear.csv")
    retrieval_eval_NaiveRAG_Subquery('linear', QA_groundTruth_df, RAG_results_df)
    
    
    ##############################################    
    
    ### Hybrid RAG - Converging
    QA_groundTruth_df = pd.read_csv("../data/QA_Converging.csv")
    RAG_results_df = pd.read_csv("results/HybridRAG_Converging.csv")
    retrieval_eval_HybridRAG('converging', QA_groundTruth_df, RAG_results_df)
    
    ### Hybrid RAG - Divergent
    QA_groundTruth_df = pd.read_csv("../data/QA_Divergent.csv")
    RAG_results_df = pd.read_csv("results/HybridRAG_Divergent.csv")
    retrieval_eval_HybridRAG('divergent', QA_groundTruth_df, RAG_results_df)
    
    ### Hybrid RAG - Linear
    QA_groundTruth_df = pd.read_csv("../data/QA_Linear.csv")
    RAG_results_df = pd.read_csv("results/HybridRAG_Linear.csv")
    retrieval_eval_HybridRAG('linear', QA_groundTruth_df, RAG_results_df)
    
    
    ##############################################    
    
    ### Rerank RAG - Converging
    QA_groundTruth_df = pd.read_csv("../data/QA_Converging.csv")
    RAG_results_df = pd.read_csv("results/RerankRAG_Converging.csv")
    retrieval_eval_RerankRAG('converging', QA_groundTruth_df, RAG_results_df)
    
    ### Rerank RAG - Divergent
    QA_groundTruth_df = pd.read_csv("../data/QA_Divergent.csv")
    RAG_results_df = pd.read_csv("results/RerankRAG_Divergent.csv")
    retrieval_eval_RerankRAG('divergent', QA_groundTruth_df, RAG_results_df)
    
    ### Rerank RAG - Linear
    QA_groundTruth_df = pd.read_csv("../data/QA_Linear.csv")
    RAG_results_df = pd.read_csv("results/RerankRAG_Linear.csv")
    retrieval_eval_RerankRAG('linear', QA_groundTruth_df, RAG_results_df)
    
    
    ##############################################    
    
    ### Naive GraphRAG - Converging
    QA_groundTruth_df = pd.read_csv("../data/QA_Converging.csv")
    RAG_results_df = pd.read_csv("results/Naive_GraphRAG_Converging.csv")
    retrieval_eval_Naive_GraphRAG('converging', QA_groundTruth_df, RAG_results_df)
    
    ### Naive GraphRAG - Divergent
    QA_groundTruth_df = pd.read_csv("../data/QA_Divergent.csv")
    RAG_results_df = pd.read_csv("results/Naive_GraphRAG_Divergent.csv")
    retrieval_eval_Naive_GraphRAG('divergent', QA_groundTruth_df, RAG_results_df)
    
    ### Naive GraphRAG - Linear
    QA_groundTruth_df = pd.read_csv("../data/QA_Linear.csv")
    RAG_results_df = pd.read_csv("results/Naive_GraphRAG_Linear.csv")
    retrieval_eval_Naive_GraphRAG('linear', QA_groundTruth_df, RAG_results_df)
    
    
    ##############################################    
    
    ### KG RAG - Converging
    QA_groundTruth_df = pd.read_csv("../data/QA_Converging.csv")
    RAG_results_df = pd.read_csv("results/KG_RAG_Converging.csv")
    retrieval_eval_KG_RAG('converging', QA_groundTruth_df, RAG_results_df)
    
    ### KG RAG - Divergent
    QA_groundTruth_df = pd.read_csv("../data/QA_Divergent.csv")
    RAG_results_df = pd.read_csv("results/KG_RAG_Divergent.csv")
    retrieval_eval_KG_RAG('divergent', QA_groundTruth_df, RAG_results_df)
    
    ### KG RAG - Linear
    QA_groundTruth_df = pd.read_csv("../data/QA_Linear.csv")
    RAG_results_df = pd.read_csv("results/KG_RAG_Linear.csv")
    retrieval_eval_KG_RAG('linear', QA_groundTruth_df, RAG_results_df)
    
    
    ##############################################    
    
    ### GraphTRACE - Converging
    QA_groundTruth_df = pd.read_csv("../data/QA_Converging.csv")
    RAG_results_df = pd.read_csv("results/GraphTRACE_Converging.csv")
    retrieval_eval_GraphTRACE('converging', QA_groundTruth_df, RAG_results_df)
    
    ### GraphTRACE - Divergent
    QA_groundTruth_df = pd.read_csv("../data/QA_Divergent.csv")
    RAG_results_df = pd.read_csv("results/GraphTRACE_Divergent.csv")
    retrieval_eval_GraphTRACE('divergent', QA_groundTruth_df, RAG_results_df)
    
    ### GraphTRACE - Linear
    QA_groundTruth_df = pd.read_csv("../data/QA_Linear.csv")
    RAG_results_df = pd.read_csv("results/GraphTRACE_Linear.csv")
    retrieval_eval_GraphTRACE('linear', QA_groundTruth_df, RAG_results_df)
    
    '''
