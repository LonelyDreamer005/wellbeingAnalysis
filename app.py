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


def load_data() -> pd.DataFrame:
    data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError("Could not find refined_screen_time_dataset.csv")
    return pd.read_csv(data_path)


def available_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def train_best_model(df: pd.DataFrame, input_col: str, target_col: str):
    X = df[[input_col]]
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
    numeric_cols = available_numeric_columns(df)

    prediction = None
    selected_input_col = None
    selected_col = None
    entered_value = None
    model_name = None
    model_r2 = None
    error = None

    if not numeric_cols:
        error = "No numeric target columns are available for prediction."
        return render_template(
            "index.html",
            input_cols=[],
            targets=[],
            prediction=prediction,
            selected_input_col=selected_input_col,
            selected_col=selected_col,
            entered_value=entered_value,
            model_name=model_name,
            model_r2=model_r2,
            error=error,
        )

    default_input_col = (
        "screen_time_hours" if "screen_time_hours" in numeric_cols else numeric_cols[0]
    )
    default_col = (
        "mental_wellness_index_0_100" if "mental_wellness_index_0_100" in numeric_cols else numeric_cols[0]
    )

    if request.method == "POST":
        selected_input_col = request.form.get("input_col", default_input_col)
        selected_col = request.form.get("target_col", default_col)
        entered_value = request.form.get("input_value", "").strip()

        if selected_input_col not in numeric_cols or selected_col not in numeric_cols:
            error = "Selected input or target column is not valid."
        elif selected_input_col == selected_col:
            error = "Input column and target column must be different."
        else:
            try:
                input_value = float(entered_value)
                model, model_name, model_r2 = train_best_model(
                    df, selected_input_col, selected_col
                )
                prediction = float(
                    model.predict(pd.DataFrame({selected_input_col: [input_value]}))[0]
                )
            except ValueError:
                error = "Please enter a valid number for input value."

    active_input_col = selected_input_col or default_input_col
    active_target_col = selected_col or default_col

    scatter_points = (
        df[[active_input_col, active_target_col]]
        .dropna()
        .rename(columns={active_input_col: "x", active_target_col: "y"})
        .to_dict(orient="records")
    )

    distribution_cols = [
        col
        for col in [
            "screen_time_hours",
            "sleep_hours",
            "stress_level_0_10",
            "mental_wellness_index_0_100",
        ]
        if col in numeric_cols
    ]
    if not distribution_cols:
        distribution_cols = numeric_cols[: min(4, len(numeric_cols))]

    distribution_data = {
        col: df[col].dropna().tolist() for col in distribution_cols
    }

    corr_df = df[numeric_cols].corr().fillna(0).round(3)
    corr_labels = corr_df.columns.tolist()
    corr_values = corr_df.values.tolist()
    corr_rows = [
        {"feature": row_name, "corr_values": row_values}
        for row_name, row_values in zip(corr_labels, corr_values)
    ]

    selected_pair_corr = 0.0
    if active_input_col in df.columns and active_target_col in df.columns:
        selected_pair_corr = float(
            df[[active_input_col, active_target_col]].corr().iloc[0, 1]
        )

    return render_template(
        "index.html",
        input_cols=numeric_cols,
        targets=numeric_cols,
        prediction=prediction,
        selected_input_col=active_input_col,
        selected_col=active_target_col,
        entered_value=entered_value,
        model_name=model_name,
        model_r2=model_r2,
        error=error,
        scatter_points=scatter_points,
        corr_labels=corr_labels,
        corr_values=corr_values,
        corr_rows=corr_rows,
        distribution_data=distribution_data,
        row_count=len(df),
        numeric_count=len(numeric_cols),
        selected_pair_corr=selected_pair_corr,
    )


if __name__ == "__main__":
    app.run(debug=True)
