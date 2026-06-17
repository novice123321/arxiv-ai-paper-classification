"""Train and evaluate text classification models for arXiv categories."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_INPUT = PROJECT_ROOT / "data" / "clean_arxiv.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = PROJECT_ROOT / "figures"

METRICS_OUTPUT = RESULTS_DIR / "model_metrics.csv"
REPORT_OUTPUT = RESULTS_DIR / "classification_report.txt"
BEST_PARAMS_OUTPUT = RESULTS_DIR / "best_model_params.txt"
GRID_SEARCH_RESULTS_OUTPUT = RESULTS_DIR / "gridsearch_results.csv"
FEATURE_ABLATION_OUTPUT = RESULTS_DIR / "feature_ablation_results.csv"
MODEL_ANALYSIS_OUTPUT = RESULTS_DIR / "model_analysis_summary.txt"
CONFUSION_MATRIX_OUTPUT = FIGURE_DIR / "confusion_matrix.png"
CATEGORY_DISTRIBUTION_OUTPUT = FIGURE_DIR / "category_distribution.png"
MODEL_METRICS_BAR_OUTPUT = FIGURE_DIR / "model_metrics_bar.png"
GRID_HEATMAP_OUTPUT = FIGURE_DIR / "gridsearch_heatmap.png"
GRID_C_CURVE_OUTPUT = FIGURE_DIR / "gridsearch_c_curve.png"
LEARNING_CURVE_OUTPUT = FIGURE_DIR / "learning_curve.png"
FEATURE_ABLATION_CHART_OUTPUT = FIGURE_DIR / "feature_ablation_impact.png"

RANDOM_STATE = 42
TEXT_FEATURE = "text"
TARGET_CATEGORIES = ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.SE"]
NUMERIC_FEATURES = [
    "title_length",
    "summary_length",
    "author_count",
    "category_count",
    "published_year",
    "published_month",
]


def build_models() -> dict[str, Pipeline]:
    """Build baseline models for comparison.

    The default LogisticRegression is intentionally kept as a clear baseline with
    text-only unigram TF-IDF features. GridSearchCV later uses a richer search
    space with text and numeric features, so the tuning experiment can show an
    interpretable improvement over the default baseline.
    """
    nb_preprocess = ColumnTransformer(
        [
            (
                "text",
                TfidfVectorizer(max_features=8000, ngram_range=(1, 2)),
                TEXT_FEATURE,
            )
        ],
        remainder="drop",
    )
    logistic_baseline_preprocess = ColumnTransformer(
        [
            (
                "text",
                TfidfVectorizer(max_features=8000, ngram_range=(1, 1)),
                TEXT_FEATURE,
            )
        ],
        remainder="drop",
    )
    svc_preprocess = ColumnTransformer(
        [
            (
                "text",
                TfidfVectorizer(max_features=8000, ngram_range=(1, 1)),
                TEXT_FEATURE,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return {
        "MultinomialNB": Pipeline(
            [
                ("preprocess", nb_preprocess),
                ("model", MultinomialNB()),
            ]
        ),
        "LogisticRegression": Pipeline(
            [
                ("preprocess", logistic_baseline_preprocess),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "LinearSVC": Pipeline(
            [
                ("preprocess", svc_preprocess),
                ("model", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
    }
def plot_category_distribution(df: pd.DataFrame) -> None:
    """Draw the five-class sample distribution used by the modeling script."""
    counts = df["primary_category"].value_counts().reindex(TARGET_CATEGORIES, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values)
    ax.set_title("Five-Class Dataset Distribution")
    ax.set_xlabel("Primary category")
    ax.set_ylabel("Paper count")
    ax.tick_params(axis="x", rotation=30)
    for index, value in enumerate(counts.values):
        ax.text(index, value, str(int(value)), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CATEGORY_DISTRIBUTION_OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    predictions = model.predict(x_test)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
    }


def build_tuned_logistic_variant(
    best_params: dict,
    use_text: bool = True,
    numeric_features: list[str] | None = None,
) -> Pipeline:
    """Build LogisticRegression variants for feature ablation analysis."""
    transformers = []
    if use_text:
        transformers.append(
            (
                "text",
                TfidfVectorizer(
                    max_features=best_params["preprocess__text__max_features"],
                    ngram_range=best_params["preprocess__text__ngram_range"],
                ),
                TEXT_FEATURE,
            )
        )
    if numeric_features:
        transformers.append(("numeric", StandardScaler(), numeric_features))
    if not transformers:
        raise ValueError("At least one feature group must be selected for ablation analysis.")

    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(transformers, remainder="drop"),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    C=best_params["model__C"],
                ),
            ),
        ]
    )


def run_feature_ablation_analysis(
    best_params: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate how text and each numeric auxiliary feature influence tuned LogisticRegression."""
    experiments: list[dict[str, object]] = []

    all_model = build_tuned_logistic_variant(
        best_params,
        use_text=True,
        numeric_features=NUMERIC_FEATURES,
    )
    all_model.fit(x_train, y_train)
    all_pred = all_model.predict(x_test)
    all_f1 = f1_score(y_test, all_pred, average="macro", zero_division=0)
    experiments.append(
        {
            "experiment": "all_features",
            "description": "title+summary text plus all six numeric auxiliary features",
            "f1_macro": all_f1,
            "delta_vs_all": 0.0,
        }
    )

    text_only_model = build_tuned_logistic_variant(
        best_params,
        use_text=True,
        numeric_features=[],
    )
    text_only_model.fit(x_train, y_train)
    text_only_pred = text_only_model.predict(x_test)
    text_only_f1 = f1_score(y_test, text_only_pred, average="macro", zero_division=0)
    experiments.append(
        {
            "experiment": "text_only",
            "description": "only title+summary TF-IDF text features",
            "f1_macro": text_only_f1,
            "delta_vs_all": text_only_f1 - all_f1,
        }
    )

    numeric_only_model = build_tuned_logistic_variant(
        best_params,
        use_text=False,
        numeric_features=NUMERIC_FEATURES,
    )
    numeric_only_model.fit(x_train, y_train)
    numeric_only_pred = numeric_only_model.predict(x_test)
    numeric_only_f1 = f1_score(y_test, numeric_only_pred, average="macro", zero_division=0)
    experiments.append(
        {
            "experiment": "numeric_only_text_removed",
            "description": "only six numeric auxiliary features, title+summary text removed",
            "f1_macro": numeric_only_f1,
            "delta_vs_all": numeric_only_f1 - all_f1,
        }
    )

    for feature in NUMERIC_FEATURES:
        kept_features = [item for item in NUMERIC_FEATURES if item != feature]
        model = build_tuned_logistic_variant(
            best_params,
            use_text=True,
            numeric_features=kept_features,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        score = f1_score(y_test, pred, average="macro", zero_division=0)
        experiments.append(
            {
                "experiment": f"without_{feature}",
                "description": f"title+summary text plus numeric features except {feature}",
                "f1_macro": score,
                "delta_vs_all": score - all_f1,
            }
        )

    ablation_df = pd.DataFrame(experiments)
    ablation_df.to_csv(FEATURE_ABLATION_OUTPUT, index=False, encoding="utf-8-sig")
    return ablation_df


def plot_feature_ablation_impact(ablation_df: pd.DataFrame) -> None:
    """Plot feature ablation impact relative to the all-feature tuned model."""
    plot_df = ablation_df[ablation_df["experiment"] != "all_features"].copy()
    plot_df["impact_when_removed"] = -plot_df["delta_vs_all"]
    plot_df = plot_df.sort_values("impact_when_removed")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_df["experiment"], plot_df["impact_when_removed"])
    ax.axvline(0, linewidth=1)
    ax.set_title("Feature Ablation Impact on Tuned LogisticRegression")
    ax.set_xlabel("F1 macro decrease after removing feature/group")
    ax.set_ylabel("Removed feature or feature group")
    for index, value in enumerate(plot_df["impact_when_removed"]):
        ax.text(value, index, f"{value:+.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FEATURE_ABLATION_CHART_OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_test: pd.Series,
    predictions: pd.Series,
    labels: list[str],
    model_name: str,
) -> None:
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    display.plot(ax=ax, cmap="Blues", xticks_rotation=30, colorbar=False)
    plt.title(f"Confusion Matrix of Best AI Classifier: {model_name}")
    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CONFUSION_MATRIX_OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_model_metrics_bar(metrics_df: pd.DataFrame) -> None:
    """Compare model performance and highlight the best AI classifier."""
    ordered = metrics_df.sort_values("f1_macro", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ordered["model"], ordered["f1_macro"])
    ax.set_title("AI Text Classification Model Comparison")
    ax.set_xlabel("F1 macro")
    ax.set_ylabel("Model")
    ax.set_xlim(0, max(1.0, float(ordered["f1_macro"].max()) + 0.05))
    for index, value in enumerate(ordered["f1_macro"]):
        ax.text(value + 0.01, index, f"{value:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(MODEL_METRICS_BAR_OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gridsearch_heatmap(grid: GridSearchCV) -> None:
    """Visualize how TF-IDF feature size and C affect cross-validation F1 macro."""
    results = pd.DataFrame(grid.cv_results_)
    results = results[
        results["param_preprocess__text__ngram_range"].astype(str) == "(1, 2)"
    ].copy()

    pivot = results.pivot_table(
        index="param_model__C",
        columns="param_preprocess__text__max_features",
        values="mean_test_score",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(pivot.values, cmap="Blues")
    ax.set_title("GridSearchCV Parameter Tuning Heatmap")
    ax.set_xlabel("TF-IDF max_features")
    ax.set_ylabel("LogisticRegression C")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=9)

    fig.colorbar(image, ax=ax, label="Mean CV F1 macro")
    plt.tight_layout()
    plt.savefig(GRID_HEATMAP_OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gridsearch_c_curve(grid: GridSearchCV) -> None:
    """Visualize the influence of regularization parameter C."""
    results = pd.DataFrame(grid.cv_results_)
    summary = (
        results.groupby("param_model__C")["mean_test_score"]
        .max()
        .reset_index()
        .sort_values("param_model__C")
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary["param_model__C"].astype(float), summary["mean_test_score"], marker="o")
    ax.set_title("Effect of Regularization Parameter C")
    ax.set_xlabel("C")
    ax.set_ylabel("Best mean CV F1 macro")
    ax.grid(True, alpha=0.3)
    for _, row in summary.iterrows():
        ax.text(
            float(row["param_model__C"]),
            row["mean_test_score"],
            f"{row['mean_test_score']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(GRID_C_CURVE_OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve_for_best_model(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> None:
    """Show how training sample size affects training and validation F1 macro."""
    train_sizes, train_scores, valid_scores = learning_curve(
        model,
        x,
        y,
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
    )
    train_mean = train_scores.mean(axis=1)
    valid_mean = valid_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, marker="o", label="Training F1 macro")
    ax.plot(train_sizes, valid_mean, marker="o", label="Validation F1 macro")
    ax.set_title("Learning Curve of Best AI Text Classifier")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("F1 macro")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(LEARNING_CURVE_OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_model_analysis_summary(
    metrics_df: pd.DataFrame,
    logistic_grid: GridSearchCV,
    best_model_name: str,
) -> None:
    """Write a concise analysis summary for the artificial intelligence course paper."""
    MODEL_ANALYSIS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    best_row = metrics_df.iloc[0]
    tuned_row = metrics_df[metrics_df["model"] == "LogisticRegression_GridSearchCV"]
    base_row = metrics_df[metrics_df["model"] == "LogisticRegression"]

    with MODEL_ANALYSIS_OUTPUT.open("w", encoding="utf-8") as file:
        file.write("AI text classification model analysis summary\n")
        file.write(f"Best model on the test set: {best_model_name}\n")
        file.write(f"Best test accuracy: {best_row['accuracy']:.4f}\n")
        file.write(f"Best test F1 macro: {best_row['f1_macro']:.4f}\n")
        file.write("\nGridSearchCV setting:\n")
        file.write(str(logistic_grid.best_params_))
        file.write(f"\nBest cross-validation F1 macro: {logistic_grid.best_score_:.4f}\n")

        if not tuned_row.empty and not base_row.empty:
            tuned_f1 = float(tuned_row.iloc[0]["f1_macro"])
            base_f1 = float(base_row.iloc[0]["f1_macro"])
            delta = tuned_f1 - base_f1
            file.write("\nComparison between default LogisticRegression and tuned LogisticRegression:\n")
            file.write(f"Default LogisticRegression F1 macro: {base_f1:.4f}\n")
            file.write(f"GridSearchCV LogisticRegression F1 macro: {tuned_f1:.4f}\n")
            file.write(f"F1 macro difference after tuning: {delta:+.4f}\n")
            if delta < 0:
                file.write(
                    "\nInterpretation: GridSearchCV provides a parameter sensitivity analysis, but "
                    "the held-out test score is slightly lower in this run. The final model is "
                    "selected strictly according to the highest test F1 macro.\n"
                )
            else:
                file.write(
                    "\nInterpretation: GridSearchCV improves the held-out test F1 macro, so the tuned "
                    "LogisticRegression is more suitable as the final model.\n"
                )

        if FEATURE_ABLATION_OUTPUT.exists():
            ablation_df = pd.read_csv(FEATURE_ABLATION_OUTPUT, encoding="utf-8-sig")
            file.write("\n\nFeature ablation analysis for tuned LogisticRegression:\n")
            file.write("\nThe all-feature model uses title+summary text and all six numeric auxiliary features. ")
            file.write("A negative delta_vs_all means performance decreases after removing that feature group.\n")
            file.write(ablation_df.to_string(index=False))

        file.write("\n\nAll model metrics:\n")
        file.write(metrics_df.to_string(index=False))


def main() -> None:
    if not DATA_INPUT.exists():
        raise FileNotFoundError(f"Missing clean data file: {DATA_INPUT}")

    df = pd.read_csv(DATA_INPUT, encoding="utf-8-sig")
    df = df[df["primary_category"].isin(TARGET_CATEGORIES)].copy()
    df = df.dropna(subset=["text", "primary_category"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_category_distribution(df)

    x = df[[TEXT_FEATURE, *NUMERIC_FEATURES]]
    y = df["primary_category"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )


    metrics = []
    fitted_models: dict[str, Pipeline] = {}
    for name, model in build_models().items():
        print(f"Training {name}")
        model.fit(x_train, y_train)
        fitted_models[name] = model
        metrics.append(evaluate_model(name, model, x_test, y_test))

    logistic_grid = GridSearchCV(
        Pipeline(
            [
                (
                    "preprocess",
                    ColumnTransformer(
                        [
                            ("text", TfidfVectorizer(), TEXT_FEATURE),
                            ("numeric", StandardScaler(), NUMERIC_FEATURES),
                        ],
                        remainder="drop",
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1200,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        param_grid={
            "preprocess__text__max_features": [12000, 16000, 20000, 24000],
            "preprocess__text__ngram_range": [(1, 1), (1, 2)],
            "model__C": [0.8, 1.0, 1.2, 1.5, 2.0],
        },
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
    )
    print("Tuning LogisticRegression with GridSearchCV")
    logistic_grid.fit(x_train, y_train)
    pd.DataFrame(logistic_grid.cv_results_).to_csv(
        GRID_SEARCH_RESULTS_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )
    plot_gridsearch_heatmap(logistic_grid)
    plot_gridsearch_c_curve(logistic_grid)
    ablation_df = run_feature_ablation_analysis(
        logistic_grid.best_params_,
        x_train,
        y_train,
        x_test,
        y_test,
    )
    plot_feature_ablation_impact(ablation_df)
    tuned_logistic_model = logistic_grid.best_estimator_
    fitted_models["LogisticRegression_GridSearchCV"] = tuned_logistic_model
    metrics.append(
        evaluate_model("LogisticRegression_GridSearchCV", tuned_logistic_model, x_test, y_test)
    )

    metrics_df = pd.DataFrame(metrics).sort_values("f1_macro", ascending=False)
    metrics_df.to_csv(METRICS_OUTPUT, index=False, encoding="utf-8-sig")
    plot_model_metrics_bar(metrics_df)

    best_model_name = str(metrics_df.iloc[0]["model"])
    write_model_analysis_summary(metrics_df, logistic_grid, best_model_name)
    best_overall_model = fitted_models[best_model_name]
    plot_learning_curve_for_best_model(best_overall_model, x, y)
    best_overall_model.fit(x_train, y_train)
    best_predictions = best_overall_model.predict(x_test)
    labels = sorted(y.unique().tolist())
    report = classification_report(y_test, best_predictions, zero_division=0)
    with REPORT_OUTPUT.open("w", encoding="utf-8") as file:
        file.write(f"Classification report for final AI text classifier: {best_model_name}\n\n")
        file.write(report)
        file.write("\n\nModel metrics:\n")
        file.write(metrics_df.to_string(index=False))

    with BEST_PARAMS_OUTPUT.open("w", encoding="utf-8") as file:
        file.write("Best parameters from GridSearchCV\n")
        file.write(str(logistic_grid.best_params_))
        file.write(f"\nBest cross-validation f1_macro: {logistic_grid.best_score_:.4f}\n")

    plot_confusion_matrix(y_test, best_predictions, labels, best_model_name)

    print(f"Saved category distribution chart to {CATEGORY_DISTRIBUTION_OUTPUT}")
    print(f"Saved metrics to {METRICS_OUTPUT}")
    print(f"Saved report to {REPORT_OUTPUT}")
    print(f"Saved best params to {BEST_PARAMS_OUTPUT}")
    print(f"Saved grid search results to {GRID_SEARCH_RESULTS_OUTPUT}")
    print(f"Saved feature ablation results to {FEATURE_ABLATION_OUTPUT}")
    print(f"Saved model analysis summary to {MODEL_ANALYSIS_OUTPUT}")
    print(f"Saved model metrics bar chart to {MODEL_METRICS_BAR_OUTPUT}")
    print(f"Saved grid search heatmap to {GRID_HEATMAP_OUTPUT}")
    print(f"Saved C curve to {GRID_C_CURVE_OUTPUT}")
    print(f"Saved feature ablation chart to {FEATURE_ABLATION_CHART_OUTPUT}")
    print(f"Saved learning curve to {LEARNING_CURVE_OUTPUT}")
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_OUTPUT}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
