import sys
import pandas as pd

input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv(input_file, encoding="utf-8")

col_sr_dict = {
    col: df[col].str.replace("\n", " ") if df[col].dtype == "object" else df[col] for col in df.columns
}
df2 = pd.DataFrame(col_sr_dict)
df2.to_csv(output_file, encoding="utf-8", index=False)
