import pandas as pd
import re

def extract_triples(path_str):
    """
    Extrahiert Tripel (start_id, relation, end_id) aus dem Subgraph-Pfadstring.
    """
    # Finde alle Knoten
    #nodes = re.findall(r'\(:.*?id: (e\d+).*?\)', path_str)
    nodes = re.findall(r'\(:[^\)]*?id:\s*([^\}\s]+)', path_str)
    
    # Finde alle Relationen
    rels = re.findall(r'-\[:(.*?)\]->', path_str)
    
    # Baue Tripel (start, relation, end)
    triples = []
    for i in range(len(rels)):
        triple = (nodes[i], rels[i], nodes[i+1])
        triples.append(triple)
    return triples

def get_unique_triple_count(subgraph1, subgraph2):
    triples1 = extract_triples(subgraph1)
    triples2 = extract_triples(subgraph2)
    unique_triples = set(triples1 + triples2)
    return len(unique_triples)



def extract_triples_from_path(path_str):
    # Flexibler Regex für Knoten-IDs (auch bn15, etc.)
    nodes = re.findall(r'\(:[^\)]*?id:\s*([^\}\s]+)', path_str)
    
    # Relationstypen extrahieren
    rels = re.findall(r'-\[:(.*?)\]->', path_str)

    triples = []
    for i in range(len(rels)):
        triple = (nodes[i].strip(), rels[i].strip(), nodes[i+1].strip())
        triples.append(triple)
    return triples


def get_hop_count_from_path(path_str):
    triples = extract_triples_from_path(path_str)
    return len(set(triples))


if __name__ == "__main__":
    # Beispielanwendung
    #(:_2022 {id: e404})-[:has_inflation_prediction]->(:_1_point_2percent {id: e635})-[:is_inflation_rate_for]->(:euro_area {id: e375}),(:_2022 {id: e404})-[:projection_entity]->(:_1_point_4percent {id: e348})-[:is_inflation_rate_for]->(:euro_area {id: e375}),4
    #(:_2022 {id: e404})-[:uncertainty_by]->(:Russias_invasion_of_Ukraine {id: e21950})-[:caused_surge_in]->(:energy_prices {id: e3312}),(:_2022 {id: e404})-[:uncertainty_by]->(:Russias_invasion_of_Ukraine {id: e21950})-[:triggered]->(:uncertainty {id: e751})-[:PLACEHOLDER]->(:bn15 {id: bn15})-[:compare]->(:firms {id: e232})-[:associated_with_change_in]->(:selling_prices {id: e4263})-[:associated_with]->(:intermediate_goods_sector {id: e6048})-[:fall_in_activity_led_to_stagnation_in]->(:Euro_area_industrial_production {id: e22016})-[:stagnated_due_to]->(:energy_prices {id: e3312}),9
    #subgraph1 = "(:_2022 {id: e404})-[:uncertainty_by]->(:Russias_invasion_of_Ukraine {id: e21950})-[:caused_surge_in]->(:energy_prices {id: e3312})"
    #subgraph2 = "(:_2022 {id: e404})-[:uncertainty_by]->(:Russias_invasion_of_Ukraine {id: e21950})-[:triggered]->(:uncertainty {id: e751})-[:PLACEHOLDER]->(:bn15 {id: bn15})-[:compare]->(:firms {id: e232})-[:associated_with_change_in]->(:selling_prices {id: e4263})-[:associated_with]->(:intermediate_goods_sector {id: e6048})-[:fall_in_activity_led_to_stagnation_in]->(:Euro_area_industrial_production {id: e22016})-[:stagnated_due_to]->(:energy_prices {id: e3312})"
    subgraph1 = "(:_2022 {id: e404})-[:has_inflation_prediction]->(:_1_point_2percent {id: e635})-[:is_inflation_rate_for]->(:euro_area {id: e375})"
    subgraph2 = "(:_2022 {id: e404})-[:projection_entity]->(:_1_point_4percent {id: e348})-[:is_inflation_rate_for]->(:euro_area {id: e375})"
    subgraph1 = "(:Governing_Council {id: e150})-[:has_view_on]->(:inflation {id: e108})-[:due_to]->(:anchored_inflation_expectations {id: e18758})"
    subgraph2 = "(:Governing_Council {id: e150})-[:has_view_on]->(:inflation {id: e108})-[:is]->(:high_and_volatile_inflation {id: e16498})"
    
    count = get_unique_triple_count(subgraph1, subgraph2)
    print(f"Anzahl der einzigartigen Tripel: {count}")
    
    
    path = "(:China {id: e207})-[:is_being_discussed_concerning]->(:real_GDP_growth {id: e3611})-[:outlook_for_in]->(:Turkey {id: e550})-[:InflationRateIn]->(:January {id: e278})-[:has_less_growth_rate_than]->(:February {id: e1135})-[:inflation_rate_for_month]->(:inflation_rate {id: e3512})-[:rate_in_percentage_in_february]->(:_5_point_9percent {id: e9172})"
    path = "(:_2022 {id: e404})-[:uncertainty_by]->(:Russias_invasion_of_Ukraine {id: e21950})-[:caused_surge_in]->(:energy_prices {id: e3312})-[:PLACEHOLDER]->(:bn15 {id: bn15})-[:PLACEHOLDER]->(:government {id: e1782})-[:included_in_economic_policy_responses]->(:Term_Funding_Scheme {id: e14464})"
    count = get_hop_count_from_path(path)
    print("Hop Count Path:", count)


#df = pd.read_csv("../data/QA_small/QA_divergent_with_blank_from_Subgraph_with_path_validation.csv")

# Neue Spalte mit Hop Counts berechnen
#df["hop_count"] = df.apply(lambda row: get_unique_triple_count(str(row["Subgraph1"]), str(row["Subgraph2"])), axis=1)

#df.to_csv("../data/QA_small/QA_Divergent_small.csv", index=False)

df = pd.read_csv("../data/QA_small/QA_linear_with_blank_from_Subgraph_with_path_validation.csv")
df["hop_count"] = df.apply(lambda row: get_hop_count_from_path(str(row["Path"])), axis=1)
#df.to_csv("../data/QA_small/QA_Linear_small.csv", index=False)

