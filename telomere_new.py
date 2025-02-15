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
    df = df.drop(
        columns=[
            "socioeconomic_status",
            "bp",
            "bmi_category",
            "hr_category",
            "rr_category",
            "cardiovascular_disease_diagnosis",
            "allergy_disease_diagnosis",
        ]
    )  # find a way to include?
    return df


def preprocess_data(df):
    cigarette_smoking_mapping = {
        "No information": np.nan,
        "Former Smoker": np.nan,
        "Never Smoker": 0,
        "Occasional Smoker": 1,
        "Regular Smoker": 2,
    }

    df["cigarette_smoking"] = df["cigarette_smoking"].replace(cigarette_smoking_mapping)

    physical_activity_mapping = {
        "No information": np.nan,
        "Sedentary (Inactive)": 0,
        "Minimally Active": 1,
        "Lightly Active": 2,
        "Moderately Active": 3,
        "Highly Active": 4,
    }

    df["physical_activity_cohort"] = df["physical_activity_cohort"].replace(
        physical_activity_mapping
    )

    alcohol_drinking_mapping = {
        "No information": np.nan,
        "Former Drinker": np.nan,
        "Never Drinker": 0,
        "Occasional Drinker": 1,
        "Moderate Drinker": 2,
        "Heavy Drinker": 3,
    }

    df["alcohol_drinking"] = df["alcohol_drinking"].replace(alcohol_drinking_mapping)

    education_cohort_mapping = {
        "No information": np.nan,
        "Elementary Graduate": 0,
        "High School Graduate": 1,
        "College Undergraduate": 2,
        "Vocational Graduate": 3,
        "College Graduate": 4,
        "Postgraduate (Master's or Doctorate)": 5,
    }

    df["education_cohort"] = df["education_cohort"].replace(education_cohort_mapping)

    bp_category_mapping = {
        "No information": np.nan,
        "Hypotension (Low BP)": 0,
        "Normal BP": 1,
        "Elevated BP": 2,
        "Hypertension Stage 1": 3,
        "Hypertension Stage 2": 4,
        "Hypertensive Crisis": 5,
    }

    df["bp_category"] = df["bp_category"].replace(bp_category_mapping)

    health_condition_mapping = {
        "^Clinically healthy*": 0,
        "^Single.*": 1,
        "^Multi.*": 2,
    }
    df["health_condition"] = df["health_condition"].replace(
        health_condition_mapping, regex=True
    )

    df["hr"] = pd.to_numeric(df["hr"], errors="coerce")
    df["rr"] = pd.to_numeric(df["rr"], errors="coerce")

    df["hr"] = df["hr"].fillna(df["hr"].median())
    df["rr"] = df["rr"].fillna(df["rr"].median())
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    df["education_cohort"] = df["education_cohort"].fillna(
        df["education_cohort"].mode()[0]
    )

    df["alcohol_drinking"] = df["alcohol_drinking"].fillna(
        df["alcohol_drinking"].mode()[0]
    )

    df["cigarette_smoking"] = df["cigarette_smoking"].fillna(
        df["cigarette_smoking"].mode()[0]
    )

    df["bp_category"] = df["bp_category"].fillna(df["bp_category"].mode()[0])

    df["physical_activity_cohort"] = df["physical_activity_cohort"].fillna(
        df["physical_activity_cohort"].mode()[0]
    )

    df = pd.get_dummies(df, columns=["rural_or_urban", "sex", "marital_status"])

    return df


def train_model(df):
    y = df.pop("telomere_length")
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    model = make_pipeline(StandardScaler(), SGDRegressor())

    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    comparison = pd.DataFrame({"Dataset": y_test, "Predicted": y_test_pred})
    print(comparison.head(20))

    print("R^2 score:", model.score(X_test, y_test))
    print("MSE:", mean_squared_error(y_test, y_test_pred))
    print("MAE:", mean_absolute_error(y_test, y_test_pred))


if __name__ == "__main__":
    df = read_data()
    df_preprocessed = preprocess_data(df)
    train_model(df_preprocessed)

    # print(df_preprocessed.isnull().sum().sum())
    # df_preprocessed.to_csv("telomere_preprocessed.csv", index=None)
    # df_preprocessed[df.isna().any(axis=1)].to_csv("na_data.csv", index=None)
