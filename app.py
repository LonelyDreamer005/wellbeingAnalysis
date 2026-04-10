from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

DATA_CANDIDATES = [
    Path("data/refined_screen_time_dataset.csv"),
    Path("refined_screen_time_dataset.csv"),
]
FEATURE_COL = "screen_time_hours"
EXCLUDED_TARGETS = {FEATURE_COL, "screen_time_check", "screen_time_diff"}


def load_data() -> pd.DataFrame:
    data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError("Could not find refined_screen_time_dataset.csv")
    return pd.read_csv(data_path)


def available_target_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return [c for c in numeric_cols if c not in EXCLUDED_TARGETS]


def train_best_model(df: pd.DataFrame, target_col: str):
    X = df[[FEATURE_COL]]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200, random_state=42
        ),
    }

    best_name = None
    best_model = None
    best_r2 = float("-inf")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        if score > best_r2:
            best_r2 = score
            best_name = name
            best_model = model

    return best_model, best_name, best_r2


@app.route("/", methods=["GET", "POST"])
def index():
    df = load_data()
    targets = available_target_columns(df)

    prediction = None
    selected_col = None
    entered_hours = None
    model_name = None
    model_r2 = None
    error = None

    if not targets:
        error = "No numeric target columns are available for prediction."
        return render_template(
            "index.html",
            targets=[],
            prediction=prediction,
            selected_col=selected_col,
            entered_hours=entered_hours,
            model_name=model_name,
            model_r2=model_r2,
            error=error,
        )

    default_col = (
        "mental_wellness_index_0_100"
        if "mental_wellness_index_0_100" in targets
        else targets[0]
    )

    if request.method == "POST":
        selected_col = request.form.get("target_col", default_col)
        entered_hours = request.form.get("screen_time_hours", "").strip()

        if selected_col not in targets:
            error = "Selected column is not valid."
        else:
            try:
                hours_value = float(entered_hours)
                model, model_name, model_r2 = train_best_model(df, selected_col)
                prediction = float(
                    model.predict(pd.DataFrame({FEATURE_COL: [hours_value]}))[0]
                )
            except ValueError:
                error = "Please enter a valid number for screen time hours."

    return render_template(
        "index.html",
        targets=targets,
        prediction=prediction,
        selected_col=selected_col or default_col,
        entered_hours=entered_hours,
        model_name=model_name,
        model_r2=model_r2,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
