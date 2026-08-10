import pandas as pd # helps load in the csv file as we used that to store the data in the previous step
data=pd.read_csv("data/interim/reports_flat.csv") # loads in the csv file as a pandas dataframe
print("rows before cleaning",len(data)) # confirms the number of rows before cleaning

data["FINDINGS"]=data["FINDINGS"].fillna("") # fills in the missing values in the FINDINGS column with an empty string
data["IMPRESSION"]=data["IMPRESSION"].fillna("") # fills in the missing values in the IMPRESSION column with an empty string

data["FINDINGS"]=data["FINDINGS"].str.strip() # removing white spaces from the FINDINGS column
data["IMPRESSION"]=data["IMPRESSION"].str.strip() # removing white spaces from the IMPRESSION column

data = data[ # excludes empty rows in both columns
    (data["FINDINGS"] != "") &
    (data["IMPRESSION"] != "")
]

data= data.drop_duplicates() # gets rid of any duplicate rows in the dataframe

print("rows after cleaning",len(data)) # confirms the number of rows after cleaning

data.to_csv(
    "data/interim/reports_clean.csv",
    index=False
)

