import sys
import pandas as pd

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_csv> <output_csv>", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        df = pd.read_csv(input_file, encoding="utf-8")

        col_sr_dict = {
            col: df[col].str.replace("\n", " ") if df[col].dtype == "object" else df[col] for col in df.columns
        }
        df2 = pd.DataFrame(col_sr_dict)
        df2.to_csv(output_file, encoding="utf-8", index=False)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
