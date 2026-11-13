import pandas as pd
from pathlib import Path
data_path = Path(__file__).parent.parent / "data" / "OnlineRetail.xlsx"
df = pd.read_excel(data_path)

print("------------data overview------------")
print(df.info())
print(df.head())
print(df.tail())