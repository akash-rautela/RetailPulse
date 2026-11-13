import pandas as pd
from pathlib import Path

df = pd.read_excel("C:/Users/VICTUS/OneDrive/Desktop/Retail Pulse/data/OnlineRetail.xlsx")
#handle missing values
df.dropna(inplace=True)
print(df.isna().sum())

#handle duplicates
df.drop_duplicates(inplace=True)

#remove cancelled transactions
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

#Remove invalid quantities and prices
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

#Convert invoice date to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

#Create total transaction value
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

df.to_csv("C:/Users/VICTUS/OneDrive/Desktop/Retail Pulse/data/cleaned_data.csv", index=False)

print("\n========== CLEANING RESULTS ==========")
print("Cleaned dataset shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("\nDate range:")
print(df["InvoiceDate"].min(), "to", df["InvoiceDate"].max())

print("\nTotal customers:", df["CustomerID"].nunique())
print("Total transactions:", df["InvoiceNo"].nunique())
print("Total revenue:", round(df["TotalAmount"].sum(), 2))

print("\nCleaned dataset saved successfully!")