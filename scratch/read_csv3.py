import csv
import os

def main():
    csv_path = os.path.join("pokemon-tcg-ai-battle", "JP_Card_Data.csv")
    out_dir = "scratch"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "parsed_cards.txt")
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        
        with open(out_path, "w", encoding="utf-8") as out:
            out.write("Header: " + ",".join(header) + "\n")
            for row in reader:
                if row[0] == "21":
                    out.write("Row 21: " + ",".join(row) + "\n")
                elif row[0] == "22":
                    out.write("Row 22: " + ",".join(row) + "\n")

if __name__ == "__main__":
    main()
