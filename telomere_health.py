import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def read_data():
    df = pd.read_csv("./input/telomere_geography_health.csv")
    df = df.drop(
        columns=[
            "socioeconomic_status",
            "bp",
            "bmi_category",
            "hr_category",
            "rr_category",
            "health_condition",
        ]
    )
    return df


def categorical_to_numeric(df, column, mapping, regex=False):
    df[column] = df[column].replace(mapping, regex=regex)


def fill_na(df, column, default):
    fill_value = np.nan
    if default == "median":
        fill_value = df[column].median()
    elif default == "mode":
        fill_value = df[column].mode()[0]

    df[column] = df[column].fillna(fill_value)


def preprocess_data(df):

    categorical_to_numeric(
        df,
        "cigarette_smoking",
        {
            "No information": np.nan,
            "Former Smoker": np.nan,
            "Never Smoker": 0,
            "Occasional Smoker": 1,
            "Regular Smoker": 2,
        },
    )

    categorical_to_numeric(
        df,
        "physical_activity_cohort",
        {
            "No information": np.nan,
            "Sedentary (Inactive)": 0,
            "Minimally Active": 1,
            "Lightly Active": 2,
            "Moderately Active": 3,
            "Highly Active": 4,
        },
    )

    categorical_to_numeric(
        df,
        "alcohol_drinking",
        {
            "No information": np.nan,
            "Former Drinker": np.nan,
            "Never Drinker": 0,
            "Occasional Drinker": 1,
            "Moderate Drinker": 2,
            "Heavy Drinker": 3,
        },
    )

    categorical_to_numeric(
        df,
        "education_cohort",
        {
            "No information": np.nan,
            "Elementary Graduate": 0,
            "High School Graduate": 1,
            "College Undergraduate": 2,
            "Vocational Graduate": 3,
            "College Graduate": 4,
            "Postgraduate (Master's or Doctorate)": 5,
        },
    )

    categorical_to_numeric(
        df,
        "bp_category",
        {
            "No information": np.nan,
            "Hypotension (Low BP)": 0,
            "Normal BP": 1,
            "Elevated BP": 2,
            "Hypertension Stage 1": 3,
            "Hypertension Stage 2": 4,
            "Hypertensive Crisis": 5,
        },
    )

    categorical_to_numeric(
        df,
        "cardiovascular_disease_diagnosis",
        {
            "^No known*": 0,
            "^Non-cardiovascular*": 0,
            "^Single.*": 1,
            "^Multi.*": 1,
        },
        regex=True,
    )

    categorical_to_numeric(
        df,
        "allergy_disease_diagnosis",
        {
            "^No Diagnosed*": 0,
            "^Single.*": 1,
            "^Multi.*": 1,
        },
        regex=True,
    )

    df["hr"] = pd.to_numeric(df["hr"], errors="coerce")
    df["rr"] = pd.to_numeric(df["rr"], errors="coerce")

    fill_na(df, "hr", "median")
    fill_na(df, "rr", "median")
    fill_na(df, "bmi", "median")
    fill_na(df, "education_cohort", "mode")
    fill_na(df, "alcohol_drinking", "mode")
    fill_na(df, "cigarette_smoking", "mode")
    fill_na(df, "bp_category", "mode")
    fill_na(df, "physical_activity_cohort", "mode")

    df = pd.get_dummies(df, columns=["rural_or_urban", "sex", "marital_status"])
    df.to_csv("preprocessed.csv", index=False)
    return df


def train_model(df):
    y = df.pop("cardiovascular_disease_diagnosis")
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    selector = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))
    X_train_selected = pd.DataFrame(selector.fit_transform(X_train))
    X_test_selected = selector.transform(X_test)

    selected_features = X_train.columns[selector.get_support()]
    selected_variances = selector.variances_[selector.get_support()]

    feature_variances = pd.DataFrame(
        data={"Selected Features": selected_features, "Variances": selected_variances}
    )

    pipeline = make_pipeline(StandardScaler(), LinearSVC(tol=1e-5))
    pipeline.fit(X_train_selected, y_train)

    score = pipeline.score(X_test_selected, y_test)
    confidence_scores = pipeline.decision_function(X_test_selected)

    y_prediction = pipeline.predict(X_test_selected)
    prediction_df = pd.DataFrame(
        data={
            "Actual": y_test.reset_index(drop=True).map({1: "Present", 0: "Absent"}),
            "Predicted": pd.Series(y_prediction)
            .reset_index(drop=True)
            .map({1: "Present", 0: "Absent"}),
            "Conf. Score": confidence_scores,
            "Match": pd.Series((y_prediction == y_test).astype(bool))
            .reset_index(drop=True)
            .map({True: "Correct", False: "Incorrect"}),
        }
    )

    prediction_df["Accuracy"] = pd.Series(score)

    return (pipeline, feature_variances, prediction_df.fillna(""))


def export_model(pipeline: Pipeline) -> None:  # pickle just model or entire pipeline?
    with open("telomere_geography_health.pkl", "wb") as model_file:
        pickle.dump(pipeline, model_file)


if __name__ == "__main__":
    df = read_data()
    df_preprocessed = preprocess_data(df)
    pipeline, feature_variances, prediction_df = train_model(df_preprocessed)

    print(feature_variances)
    print(prediction_df)
    # export_model(pipeline)
