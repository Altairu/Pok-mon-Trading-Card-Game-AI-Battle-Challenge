import csv
import os

def main():
    csv_path = os.path.join("pokemon-tcg-ai-battle", "JP_Card_Data.csv")
    if not os.path.exists(csv_path):
        print("File not found")
        return
        
    with open(csv_path, "r", encoding="shift_jis") as f:
        reader = csv.reader(f)
        header = next(reader)
        print("Header:", header)
        
        # 21, 22のカードを表示
        for row in reader:
            if row[0] == "21":
                print("Row 21:", row)
            elif row[0] == "22":
                print("Row 22:", row)
            elif int(row[0]) > 25:
                break

if __name__ == "__main__":
    main()
