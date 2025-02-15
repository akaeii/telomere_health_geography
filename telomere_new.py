import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

def read_data():
    df = pd.read_csv("./input/telomere_new.csv")
    return df

def preprocess_data(df):
    pass



if __name__ == "__main__":
    print("hello world")
    df = read_data()
    print(df)

