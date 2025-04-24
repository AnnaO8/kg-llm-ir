# kg-llm-ir
Github Repository of my master's thesis "Leveraging KGs and LLMs for Optimized Information Retrieval"


Instructions to get started
1. Install Visual Studio Code
2. Install git
3. Install neo4j Desktop

Folder structure
```bash
main/
├── baselines/          # Implementations of baseline methods
│   ├── NaiveRAG.py
│   ├── NaiveRAG_Subquery.py
│   ├── RerankRAG.py
│   ├── HybridRAG.py
│   ├── KG_RAG.py
│   └── ToG.py
├── data/               # Input datasets and processed data
├── evaluations/        # Evaluation scripts and metric outputs
├── proposed_approach/  # Main implementation of the proposed method
│   └── GraphTRACE.py
├── utils/              # Utility functions and shared tools
├── .env                # Environment variables (should not be versioned)
```

### Instructions how to set up a graph data base in neo4j Desktop (on your local machine)
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




