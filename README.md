# kg-llm-ir
Github Repository of my master's thesis "Leveraging KGs and LLMs for Optimized Information Retrieval"


Instructions to get started
1. Install Visual Studio Code
2. Install git
3. Install neo4j Desktop


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

Instructions how to set up a graph data base in neo4j Desktop (on your local machine)
1. Create a new project
2. Add Economic_KG.csv into the import folder of the project
3. Use theses credentials
   URI = "bolt://localhost:7687 (you might have to change the port number in the code to the provided port number by neo4j)
   USERNAME = "neo4j"
   PASSWORD = "875421963"
4. Insert nodes
5. Insert edges
6. Check if it worked
   


