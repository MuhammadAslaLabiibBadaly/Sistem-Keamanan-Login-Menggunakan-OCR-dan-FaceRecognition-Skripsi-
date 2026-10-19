import pandas as pd
import json

file_excel = "datamahasiswa2025.xlsx"

# baca semua sheet
sheets = pd.read_excel(file_excel, sheet_name=None)

# konversi DataFrame -> dict
json_data = {}
for sheet_name, df in sheets.items():
    json_data[sheet_name] = df.to_dict(orient="records")

# simpan ke JSON
with open("datamahasiswa2025.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)

print("Semua sheet BERHASIL dikonversi ke JSON")
