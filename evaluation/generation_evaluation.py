import os
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

class answerEval(BaseModel):
    comprehensiveness_winner: int
    comprehensiveness_explanation: str
    diversity_winner: int
    diversity_explanation: str
    empowerment_winner: int
    empowerment_explanation: str
    directness_winner: int
    directness_explanation: str
    overall_winner: int
    overall_winner_explanation: str



# Approach 1: NaiveRAG
# Approach 2: NaiveRAG + Subquery
# Approach 3: HybridRAG
# Approach 4: RerankRAG
# Approach 5: Naive GraphRAG
# Approach 6: KG RAG
# Approach 7: GraphTRACE

def generation_evaluation(QA_df, approach1, approach2, approach3, approach4, approach5, approach6, approach7, output_name):
    result_df = pd.DataFrame()
    for i in range(QA_df.shape[0]):
        query = QA_df['Question'].iloc[i]
        ground_truth_answer = QA_df['Answer'].iloc[i]
        
        approaches = [approach1, approach2, approach3, approach4, approach5, approach6, approach7]
        
        queries = {approach['query'].iloc[i] for approach in approaches}
        queries.add(QA_df['Question'].iloc[i])

        if len(queries) > 1:
            print(f"Mismatch in queries at row {i}")
            break

        
            
        
        answer1 = approach1['answer'].iloc[i]
        answer2 = approach2['answer'].iloc[i]
        answer3 = approach3['answer'].iloc[i]
        answer4 = approach4['answer'].iloc[i]
        answer5 = approach5['answer'].iloc[i]
        answer6 = approach6['answer'].iloc[i]
        answer7 = approach7['answer'].iloc[i]
        
        
        system_prompt = """You are an expert tasked with evaluating several answers to the same question based on these criteria: **Comprehensiveness**, **Diversity**, **Empowerment**, and **Directness**. Consider also the given ground truth answer."""
        
        prompt_template = """You will evaluate the following answers to the same question based on these criteria: **Comprehensiveness**, **Diversity**, **Empowerment**, and **Directness**. Consider also the given ground truth answer. 

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?
        - **Directness**: How clearly and directly does the answer address the question without unnecessary information?

        For each criterion, choose the best answer (either Answer 1, Answer 2, Answer 3, ..., Answer 7) and explain why. Then, select an overall winner based on the given categories.

        Here is the question:
        {query}
        
        Here is the ground truth answer:
        {ground_truth_answer}

        Here are the answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}
        
        **Answer 3:**
        {answer3}
        
        **Answer 4:**
        {answer4}
        
        **Answer 5:**
        {answer5}
        
        **Answer 6:**
        {answer6}
        
        **Answer 7:**
        {answer7}

        Evaluate the answers using the criteria listed above and provide detailed explanations for each criterion. Only one winner should be selected for each criterion.
        For the overall winner, consider the answers' performance across all criteria.

        Output your evaluation in the provided response format:
        comprehensiveness_winner: int[1,7]
        comprehensiveness_explanation: str
        diversity_winner: int[1,7]
        diversity_explanation: str
        empowerment_winner: int[1,7]
        empowerment_explanation: str
        directness_winner: int[1,7]
        directness_explanation: str
        overall_winner: int[1,7]
        overall_winner_explanation: str
        """

        
        prompt = prompt_template.format(query=query, ground_truth_answer=ground_truth_answer, answer1=answer1, answer2=answer2, answer3=answer3, answer4=answer4, answer5=answer5, answer6=answer6, answer7=answer7)
        response = client.beta.chat.completions.parse(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model="gpt-4o-mini",
            response_format=answerEval
        )

        raw_answer = response.choices[0].message.content
        json_data = json.loads(raw_answer)
        answer = answerEval(**json_data)
        
        new_row = {'query':query,'comprehensiveness_winner':answer.comprehensiveness_winner, 'comprehensiveness_explanation': answer.comprehensiveness_explanation, 'diversity_winner': answer.diversity_winner, 'diversity_explanation': answer.diversity_explanation, 'empowerment_winner': answer.empowerment_winner, 'empowerment_explanation': answer.empowerment_explanation, 'directness_winner': answer.directness_winner, 'directness_explanation': answer.directness_explanation, 'overall_winner': answer.overall_winner, 'overall_winner_explanation': answer.overall_winner_explanation}
        result_df = result_df._append(new_row, ignore_index=True)
        print(f"\rProgress: {i+1}/{QA_df.shape[0]}", end="", flush=True)

    file_path_name = f"generation_evaluation_{output_name}.csv"
    result_df.to_csv(file_path_name, index=False, encoding='utf-8')
    
    
def calculate_winner_percentage(eval_result_df):
    approach_map = {
        1: "NaiveRAG",
        2: "NaiveRAG + Subquery",
        3: "HybridRAG",
        4: "RerankRAG",
        5: "Naive GraphRAG",
        6: "KG RAG",
        7: "GraphTRACE"
    }
    
    criteria = ["comprehensiveness", "diversity", "empowerment", "directness", "overall"]
    #results = pd.DataFrame(index=approach_map.values(), columns=criteria)
    results = pd.DataFrame(index=criteria, columns=approach_map.values())
    
    for crit in criteria:
        winner_col = f"{crit}_winner"
        counts = eval_result_df[winner_col].value_counts().sort_index()
        total = counts.sum()
        for idx, count in counts.items():
            name = approach_map.get(idx, f"None")
            results.loc[crit, name] = f"{(count / total * 100):.2f}%"
    
    print(results)
    results_numeric = results.map(lambda x: float(x.strip('%')) if isinstance(x, str) else 0.0)
    

    plt.figure(figsize=(12, 6))  # Größe anpassen
    sns.heatmap(results_numeric, annot=True, fmt=".1f", cmap="RdYlGn", cbar_kws={'label': 'Gewinnanteil (%)'})

    plt.title("Heatmap of Winner Distribution per Criterion")
    plt.xlabel("Approach")
    plt.ylabel("Criterion")
    plt.tight_layout()
    plt.show()
    


def calculate_average_winner_percentage(csv_paths):
    approach_map = {
        1: "NaiveRAG",
        2: "NaiveRAG + Subquery",
        3: "HybridRAG",
        4: "RerankRAG",
        5: "Naive GraphRAG",
        6: "KG RAG",
        7: "GraphTRACE"
    }

    criteria = ["comprehensiveness", "diversity", "empowerment", "directness", "overall"]
    all_result_dfs = []

    for path in csv_paths:
        df = pd.read_csv(path)
        result = pd.DataFrame(index=criteria, columns=approach_map.values(), dtype=float)

        for crit in criteria:
            winner_col = f"{crit}_winner"
            counts = df[winner_col].value_counts().sort_index()
            total = counts.sum()
            for idx in range(1, 8):
                name = approach_map[idx]
                count = counts.get(idx, 0)
                res_num = count / total * 100
                result.loc[crit, name] = res_num

        all_result_dfs.append(result)

    avg_result = sum(all_result_dfs) / len(all_result_dfs)

    for row in avg_result.index:
        row_sum = avg_result.loc[row].sum()
        diff = 100 - row_sum

        max_col = avg_result.loc[row].idxmax()
        avg_result.loc[row, max_col] += diff

    avg_result_formatted = avg_result.map(lambda x: f"{x:.2f}%")

    print(avg_result_formatted)
    return avg_result_formatted


if __name__ == "__main__":
    ### Converging - Average ###
    csv_paths_conv = [
    "generation_evaluation_Converging_run1.csv",
    "generation_evaluation_Converging_run2.csv",
    "generation_evaluation_Converging_run3.csv"
    ]
    print("\nGeneration Evaluation - Converging - ### Average ###")
    average_percentages = calculate_average_winner_percentage(csv_paths_conv)
    
    ### Divergent - Average ###
    csv_paths_div = [
    "generation_evaluation_Divergent_run1.csv",
    "generation_evaluation_Divergent_run2.csv",
    "generation_evaluation_Divergent_run3.csv"
    ]
    print("\nGeneration Evaluation - Divergent - ### Average ###")
    average_percentages = calculate_average_winner_percentage(csv_paths_div)
    
    ### Linear - Average ###
    csv_paths_lin = [
    "generation_evaluation_Linear_run1.csv",
    "generation_evaluation_Linear_run2.csv",
    "generation_evaluation_Linear_run3.csv"
    ]
    print("\nGeneration Evaluation - Linear - ### Average ###")
    average_percentages = calculate_average_winner_percentage(csv_paths_lin)
    
    
    ''' 
    print("\nGeneration Evaluation - Converging")
    df = pd.read_csv("generation_evaluation_Converging.csv")
    calculate_winner_percentage(df)
    
    print("\nGeneration Evaluation - Divergent")
    df = pd.read_csv("generation_evaluation_Divergent.csv")
    calculate_winner_percentage(df)
    
    print("\nGeneration Evaluation - Linear")
    df = pd.read_csv("generation_evaluation_Linear.csv")
    calculate_winner_percentage(df)
    '''
    
    
    '''
    ### Converging
    QA_df = pd.read_csv("../data/QA_Converging.csv")
    approach1 = pd.read_csv("results/NaiveRAG_Converging.csv")
    approach2 = pd.read_csv("results/NaiveRAG_Subquery_Converging.csv")
    approach3 = pd.read_csv("results/HybridRAG_Converging.csv")
    approach4 = pd.read_csv("results/RerankRAG_Converging.csv")
    approach5 = pd.read_csv("results/Naive_GraphRAG_Converging.csv")
    approach6 = pd.read_csv("results/KG_RAG_Converging.csv")
    approach7 = pd.read_csv("results/GraphTRACE_Converging.csv")
    output_name = "Converging_run1"
    generation_evaluation(QA_df=QA_df, approach1=approach1, approach2=approach2, approach3=approach3, approach4=approach4, approach5=approach5, approach6=approach6, approach7=approach7, output_name=output_name)
    
    output_name = "Converging_run2"
    generation_evaluation(QA_df=QA_df, approach1=approach1, approach2=approach2, approach3=approach3, approach4=approach4, approach5=approach5, approach6=approach6, approach7=approach7, output_name=output_name)

    output_name = "Converging_run3"
    generation_evaluation(QA_df=QA_df, approach1=approach1, approach2=approach2, approach3=approach3, approach4=approach4, approach5=approach5, approach6=approach6, approach7=approach7, output_name=output_name)

    
    
    ### Divergent
    QA_df = pd.read_csv("../data/QA_Divergent.csv")
    approach1 = pd.read_csv("results/NaiveRAG_Divergent.csv")
    approach2 = pd.read_csv("results/NaiveRAG_Subquery_Divergent.csv")
    approach3 = pd.read_csv("results/HybridRAG_Divergent.csv")
    approach4 = pd.read_csv("results/RerankRAG_Divergent.csv")
    approach5 = pd.read_csv("results/Naive_GraphRAG_Divergent.csv")
    approach6 = pd.read_csv("results/KG_RAG_Divergent.csv")
    approach7 = pd.read_csv("results/GraphTRACE_Divergent.csv")
    output_name = "Divergent_run1"
    generation_evaluation(QA_df=QA_df, approach1=approach1, approach2=approach2, approach3=approach3, approach4=approach4, approach5=approach5, approach6=approach6, approach7=approach7, output_name=output_name)
    
    output_name = "Divergent_run2"
    generation_evaluation(QA_df=QA_df, approach1=approach1, approach2=approach2, approach3=approach3, approach4=approach4, approach5=approach5, approach6=approach6, approach7=approach7, output_name=output_name)

    output_name = "Divergent_run3"
    generation_evaluation(QA_df=QA_df, approach1=approach1, approach2=approach2, approach3=approach3, approach4=approach4, approach5=approach5, approach6=approach6, approach7=approach7, output_name=output_name)

    
    
    ### Linear
    QA_df = pd.read_csv("../data/QA_Linear.csv")
    approach1 = pd.read_csv("results/NaiveRAG_Linear.csv")
    approach2 = pd.read_csv("results/NaiveRAG_Subquery_Linear.csv")
    approach3 = pd.read_csv("results/HybridRAG_Linear.csv")
    approach4 = pd.read_csv("results/RerankRAG_Linear.csv")
    approach5 = pd.read_csv("results/Naive_GraphRAG_Linear.csv")
    approach6 = pd.read_csv("results/KG_RAG_Linear.csv")
    approach7 = pd.read_csv("results/GraphTRACE_Linear.csv")
    output_name = "Linear_run1"
    generation_evaluation(QA_df=QA_df, approach1=approach1, approach2=approach2, approach3=approach3, approach4=approach4, approach5=approach5, approach6=approach6, approach7=approach7, output_name=output_name)
    
    output_name = "Linear_run2"
    generation_evaluation(QA_df=QA_df, approach1=approach1, approach2=approach2, approach3=approach3, approach4=approach4, approach5=approach5, approach6=approach6, approach7=approach7, output_name=output_name)

    output_name = "Linear_run3"
    generation_evaluation(QA_df=QA_df, approach1=approach1, approach2=approach2, approach3=approach3, approach4=approach4, approach5=approach5, approach6=approach6, approach7=approach7, output_name=output_name)

    
    
    QA_df = pd.read_csv("../data/QA_Converging.csv")
    approach1 = pd.read_csv("results/NaiveRAG_Converging.csv")
    approach2 = pd.read_csv("results/NaiveRAG_Subquery_Converging.csv")
    approach3 = pd.read_csv("results/HybridRAG_Converging.csv")
    approach4 = pd.read_csv("results/RerankRAG_Converging.csv")
    approach5 = pd.read_csv("results/Naive_GraphRAG_Converging.csv")
    approach6 = pd.read_csv("results/KG_RAG_Converging.csv")
    approach7 = pd.read_csv("results/GraphTRACE_v2_Converging.csv")
    output_name = "Converging_GraphTRACE_v2"
    generation_evaluation(QA_df=QA_df, approach1=approach1, approach2=approach2, approach3=approach3, approach4=approach4, approach5=approach5, approach6=approach6, approach7=approach7, output_name=output_name)
    '''