import os # import os allows us to read files and write files
import pandas as pd # pandas allows us to read/create tables
from sklearn.model_selection import train_test_split # from the scikit learn library this allows us to split data for training and testing

# Loading data

data=pd.read_csv("data/interim/reports_clean.csv")

# splitting the data into training and testing sets
#0.2 is 20% of data will be used for testing, therefore 80% will be trained on
#random_state=42 is essentially a consistently picks 42 random rows for testing the data.

train_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42
)

#saving the data to the processed folder as a csv file
#index makes sure that it doesnt create any extra columns

train_data.to_csv(
    "data/processed/train_data.csv",
    index=False
)

test_data.to_csv(
    "data/processed/test_data.csv",
    index=False 
)
print("Total rows", len(data)) # prints original number of rows in the data
print("Training rows", len(train_data)) # nuber of rows that will be used to train the model
print("Testing rows", len(test_data))# nuber of rows used for tetsing the model
print("Data has been split into training and testing sets and saved to the processed folder.") # ensures all the above has worked.



