import csv
from collections import Counter

def analyze_hops(filepath, name):
    hop_counter = Counter()
    try:
        with open(filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                hop = row['hop_count'].strip()
                hop_counter[hop] += 1

        print(f"\nFile: {name}")
        for hop, count in sorted(hop_counter.items(), key=lambda x: int(float(x[0]))):
            print(f"{hop} hops: {count}")

    except Exception as e:
        print(f"Error processing file {name}: {e}")

if __name__ == "__main__":
    analyze_hops(r'..\data\QA_Converging.csv', "QA_Converging")
    analyze_hops(r'..\data\QA_Divergent.csv', "QA_Divergent")
    analyze_hops(r'..\data\QA_Linear.csv', "QA_Linear")
    
    analyze_hops(r'..\data\QA_small\QA_Converging_small.csv', "QA_Converging_small")
    analyze_hops(r'..\data\QA_small\QA_Divergent_small.csv', "QA_Divergent_small")
    analyze_hops(r'..\data\QA_small\QA_Linear_small.csv', "QA_Linear_small")