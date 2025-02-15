import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def read_data() -> pd.DataFrame:
    df = pd.read_csv("telomere_geography.csv")

    return df


def preprocess_data(df: pd.DataFrame):

    y = df.pop("Telomere Length")
    X = df

    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    model = SGDRegressor()
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    comparison = pd.DataFrame({"Dataset": y_test, "Predicted": y_test_pred})
    print(comparison.head(20))

    print("R^2 score:", model.score(X_test, y_test))
    print("MSE:", mean_squared_error(y_test, y_test_pred))
    print("MAE:", mean_absolute_error(y_test, y_test_pred))


if __name__ == "__main__":
    df = read_data()
    print(df)
    X, y = preprocess_data(df)
    print(X)
    print(y)
    model = train_model(X, y)
