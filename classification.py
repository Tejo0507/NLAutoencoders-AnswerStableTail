import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common_data import build_table

parser = argparse.ArgumentParser()
parser.add_argument("--folds", type=int, default=3)
parser.add_argument("--repeats", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default="outputs/classification")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

df = build_table()
features = [c for c in df.columns if c not in ("id", "dataset", "final_correct")]
X = df[features].values.astype(float)
y = df["final_correct"].values

print(f"rows: {len(df)}  correct: {y.sum()}  incorrect: {(1 - y).sum()}")

models = {
    "Majority baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": make_pipeline(
        StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)),
    "Random forest": RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=2,
        class_weight="balanced", random_state=args.seed),
    "Gradient boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=2, random_state=args.seed),
}

cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)

scores = {name: {"accuracy": [], "f1": [], "auroc": []} for name in models}
oof_prob = {name: np.zeros(len(y)) for name in models}
oof_count = np.zeros(len(y))

for train_idx, test_idx in cv.split(X, y):
    oof_count[test_idx] += 1
    for name, model in models.items():
        model.fit(X[train_idx], y[train_idx])
        prob = model.predict_proba(X[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        scores[name]["accuracy"].append(accuracy_score(y[test_idx], pred))
        scores[name]["f1"].append(f1_score(y[test_idx], pred, zero_division=0))
        if len(set(y[test_idx])) == 2:
            scores[name]["auroc"].append(roc_auc_score(y[test_idx], prob))
        oof_prob[name][test_idx] += prob

for name in models:
    oof_prob[name] /= oof_count

rows = []
for name, s in scores.items():
    rows.append({
        "model": name,
        "accuracy": np.mean(s["accuracy"]),
        "accuracy_sd": np.std(s["accuracy"]),
        "f1": np.mean(s["f1"]),
        "auroc": np.mean(s["auroc"]),
        "auroc_sd": np.std(s["auroc"]),
    })
table = pd.DataFrame(rows).round(3)
table.to_csv(f"{args.out}/model_scores.csv", index=False)
print()
print(table.to_string(index=False))

best = table[table["model"] != "Majority baseline"].sort_values("auroc", ascending=False).iloc[0]["model"]
print(f"\nbest model by AUROC: {best}")

fig, ax = plt.subplots(figsize=(7, 4.5))
pos = np.arange(len(table))
ax.bar(pos - 0.2, table["accuracy"], 0.4, yerr=table["accuracy_sd"], label="Accuracy", capsize=3)
ax.bar(pos + 0.2, table["auroc"], 0.4, yerr=table["auroc_sd"], label="AUROC", capsize=3)
ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
ax.set_xticks(pos)
ax.set_xticklabels(table["model"], rotation=15)
ax.set_ylim(0, 1.05)
ax.set_title("Cross-validated model comparison")
ax.legend()
fig.tight_layout()
fig.savefig(f"{args.out}/model_comparison.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
for name in models:
    if name == "Majority baseline":
        continue
    fpr, tpr, _ = roc_curve(y, oof_prob[name])
    ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, oof_prob[name]):.2f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves (out-of-fold predictions)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(f"{args.out}/roc_curves.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4.5))
pred = (oof_prob[best] >= 0.5).astype(int)
ConfusionMatrixDisplay.from_predictions(
    y, pred, display_labels=["incorrect", "correct"], cmap="Blues", ax=ax)
ax.set_title(f"Confusion matrix: {best}")
fig.tight_layout()
fig.savefig(f"{args.out}/confusion_matrix.png", dpi=200)
plt.close(fig)

forest = models["Random forest"].fit(X, y)
imp = pd.Series(forest.feature_importances_, index=features).sort_values()
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.barh(imp.index, imp.values, color="#3b7ea1")
ax.set_xlabel("Impurity importance")
ax.set_title("Random forest feature importance")
fig.tight_layout()
fig.savefig(f"{args.out}/feature_importance.png", dpi=200)
plt.close(fig)
imp.sort_values(ascending=False).round(4).to_csv(f"{args.out}/feature_importance.csv", header=["importance"])

fig, ax = plt.subplots(figsize=(6, 4))
for label, colour, text in ((1, "#2a9d8f", "correct"), (0, "#e76f51", "incorrect")):
    ax.hist(df.loc[df["final_correct"] == label, "n_tokens"], bins=10, alpha=0.7, color=colour, label=text)
ax.set_xlabel("Trace length (tokens)")
ax.set_ylabel("Count")
ax.set_title("Trace length by correctness")
ax.legend()
fig.tight_layout()
fig.savefig(f"{args.out}/length_by_class.png", dpi=200)
plt.close(fig)

print(f"\nfigures and tables written to {args.out}/")
