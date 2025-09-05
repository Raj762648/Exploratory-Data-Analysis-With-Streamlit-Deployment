## Model training and evaluation
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

##Read the dataset
def read_raw_data(file_name,file_path):
    path = os.path.join(file_path,file_name)
    df = pd.read_csv(path)
    print("Data loaded")
    return df

# Save the processed data
def save_processed_data(df,file_name,file_path):
    os.makedirs(file_path,exist_ok=True)
    path = os.path.join(file_path,file_name)
    df.to_csv(path,index=False)
    print(f"Data saved")

df = read_raw_data(file_name="iris.csv",file_path="data/raw")
mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
df["Species"] = df["Species"].map(mapping)
df.drop('Id',axis=1,inplace=True)
train_df , test_df = train_test_split(df,test_size=0.2,random_state=42)

def iris_model(df):
    X = df.drop("Species",axis=1).values
    y = df['Species'].values
    model = LogisticRegression()
    model.fit(X,y)
    print("Model Trained")
    return model

if __name__=="__main__":
    save_processed_data(train_df,file_name="train.csv",file_path="data/processed")
    save_processed_data(test_df,file_name="test.csv",file_path="data/processed")
    model = iris_model(train_df)
    joblib.dump(model,"model/iris-model.pkl")
    print("Model Saved")


    

