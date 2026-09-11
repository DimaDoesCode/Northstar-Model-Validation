from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/application_train.csv"
PACKAGE = ROOT / "model/model_package"

TARGET = "TARGET"
ID = "SK_ID_CURR"
RANDOM_STATE = 42

def build_pipeline(X: pd.DataFrame) -> tuple[ColumnTransformer, LogisticRegression]:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    return preprocessor, model

def main() -> None:
    df = pd.read_csv(DATA)
    X = df.drop(columns=[TARGET, ID])
    y = df[TARGET]

    preprocessor, model = build_pipeline(X)
    X_t = preprocessor.fit_transform(X)
    model.fit(X_t, y)

    PACKAGE.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PACKAGE / "preprocessing.joblib")
    joblib.dump(model, PACKAGE / "model.joblib")
    print("Model package written to", PACKAGE)

if __name__ == "__main__":
    main()
