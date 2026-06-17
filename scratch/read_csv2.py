import csv
import os

def try_read(encoding):
    csv_path = os.path.join("pokemon-tcg-ai-battle", "JP_Card_Data.csv")
    try:
        with open(csv_path, "r", encoding=encoding, errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)
            print(f"[{encoding}] Header:", [h[:20] for h in header])
            for row in reader:
                if row[0] == "21":
                    print(f"[{encoding}] Row 21:", row)
                    break
    except Exception as e:
        print(f"[{encoding}] Failed:", e)

def main():
    try_read("utf-8-sig")
    try_read("cp932")
    try_read("utf-8")

if __name__ == "__main__":
    main()
