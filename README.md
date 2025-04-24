# kg-llm-ir
Github Repository of my master's thesis "Leveraging KGs and LLMs for Optimized Information Retrieval"


### Instructions to get started
1. Install Visual Studio Code
2. Install git
3. Install Python
4. Install neo4j Desktop
5. Clone this GitHub repository (git clone https://github.com/AnnaO8/kg-llm-ir.git)
6. Create a virtual environment in VS Code (python -m venv venv)
7. Activate virtual environment (.venv\Scripts\activate)
   (if you have security issues, you can temporarly bypass it by running "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process")
9. Install requirements (pip install -r requirements.txt)
10. Finished :)


Folder structure
```bash
kg-llm-ir/
├── baselines/          # Implementations of baseline methods
│   ├── NaiveRAG.py
│   ├── NaiveRAG_Subquery.py
│   ├── RerankRAG.py
│   ├── HybridRAG.py
│   ├── KG_RAG.py
│   └── ToG.py
│
├── data/               # .csv files of KG, QA data sets
│   ├── Economic_KG.csv
│   ├── Economic_KG_withUnterscore.csv
│   ├── Economic-KG-with-blank.csv
│   ├── QA_Converging.csv
│   ├── QA_Divergent.csv
│   ├── QA_Linear.csv
│   ├── QA_descriptive.csv
│   ├── QA_singleEntity.csv
│   ├── QA_yesNo.csv
│   └── QA_NULL.csv
│
├── evaluation/        # Evaluation scripts and metric outputs
│   ├── results/ -> here are the retrieval evaluation results located
│   ├── retrieval_evaluation.py
│   ├── generation_evaluation.py
│   ├── retrieval_evaluation_output.txt
│   ├── generation_evaluation_output.txt
│   └── generation_evaluation_xxx.csv -> here are the generation evaluation results located
│
├── proposed_approach/  # Implementation of Graph TRACE (the proposed approach)
│   └── GraphTRACE.py
│
├── qa_generation       # QA generation pipeline
│   └── qa_gen_neo4j.py
│
├── utils/              # Utility functions and shared tools
│
├── .env                # Environment variables (should not be versioned)
```

### Instructions how to set up the graph data base in neo4j Desktop (on your local machine)
1. Create a new project
2. Add Economic_KG.csv into the import folder of the project
3. Use theses credentials  
   URI = "bolt://localhost:7687 (you might have to change the port number in the code to the provided port number by neo4j)  
   USERNAME = "neo4j"  
   PASSWORD = "875421963"  
4. Insert nodes into neo4j  
   LOAD CSV WITH HEADERS FROM 'file:///Economic-KG-cleaned.csv'  
   AS row MERGE (n:Entity {name: row.n}) MERGE (m:Entity {name: row.m});
5. Insert edges into neo4j  
   LOAD CSV WITH HEADERS FROM 'file:///Economic-KG-cleaned.csv'  
   AS row  
   MATCH (a:Entity {name: row.n})  
   MATCH (b:Entity {name: row.m})  
   MERGE (a)-[r:RELATIONSHIP {type: row.r}]->(b);
6. In order to Check and visualize 100 tripels of KG  
   MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100;  


### How to reproduce the evaluations
To reproduce the evaluations for retrieval and generation results, simply run these comands in the following folder  
./evaluation/    
python retrieval_evaluation.py
python generation_evaluation.py


### How to simply look at the results  
The following documents includes the retrieval and generation results that are published in the thesis
retrieval_evaluation_output.txt  
generation_evaluation_output.txt


### How to run one of the approaches (baselines or Graph TRACE)
1. Navigate to the respective folder (/baselines/ or /proposed_approach/)
2. run "python approach.py" (e.g. python GraphTRACE.py)


### How to use OpenAI's LLM (gpt4o-mini)
The credentials are automatically used by dotenv, but initially you have to insert your private API key 
OPEN_API_KEY=...




