import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common_data import build_table, write_provenance

parser = argparse.ArgumentParser()
parser.add_argument("--folds", type=int, default=3)
parser.add_argument("--repeats", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default="outputs/regression")
parser.add_argument("--runs", nargs="*", default=None,
                    help="run ids to pool; default is every run poolable with "
                         "the largest one (see common_data.POOLING_KEYS)")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

df, provenance = build_table(with_ast=True, run_ids=args.runs)
features = [c for c in df.columns if c not in ("id", "dataset", "tail_fraction")]
X = df[features].values.astype(float)
y = df["tail_fraction"].values

print(f"corpus: {provenance['n_rows']} rows from runs {provenance['runs']}")
if provenance["excluded"]:
    print(f"excluded runs: {provenance['excluded']}")
if provenance["dropped_without_ok_tail"]:
    print(f"dropped without an ok tail: {provenance['dropped_without_ok_tail']}")
if len(df) < 2 * args.folds:
    raise SystemExit(
        f"corpus too small: {len(df)} rows for {args.folds} folds. Only "
        f"problems whose AST status is 'ok' have a tail fraction to predict."
    )

print(f"rows: {len(df)}  target mean: {y.mean():.3f}  min: {y.min():.2f}  max: {y.max():.2f}")

models = {
    "Mean baseline": DummyRegressor(strategy="mean"),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
    "Random forest": RandomForestRegressor(
        n_estimators=300, max_depth=4, min_samples_leaf=2, random_state=args.seed),
    "Gradient boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=args.seed),
}

cv = RepeatedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)

scores = {name: {"r2": [], "mae": [], "rmse": []} for name in models}
oof_pred = {name: np.zeros(len(y)) for name in models}
oof_count = np.zeros(len(y))

for train_idx, test_idx in cv.split(X):
    oof_count[test_idx] += 1
    for name, model in models.items():
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        scores[name]["r2"].append(r2_score(y[test_idx], pred))
        scores[name]["mae"].append(mean_absolute_error(y[test_idx], pred))
        scores[name]["rmse"].append(np.sqrt(mean_squared_error(y[test_idx], pred)))
        oof_pred[name][test_idx] += pred

for name in models:
    oof_pred[name] /= oof_count

rows = []
for name, s in scores.items():
    rows.append({
        "model": name,
        "r2": np.mean(s["r2"]),
        "r2_sd": np.std(s["r2"]),
        "mae": np.mean(s["mae"]),
        "mae_sd": np.std(s["mae"]),
        "rmse": np.mean(s["rmse"]),
    })
table = pd.DataFrame(rows).round(3)
table.to_csv(f"{args.out}/model_scores.csv", index=False)
print()
print(table.to_string(index=False))

best = table[table["model"] != "Mean baseline"].sort_values("mae").iloc[0]["model"]
print(f"\nbest model by MAE: {best}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
pos = np.arange(len(table))
axes[0].bar(pos, table["mae"], yerr=table["mae_sd"], capsize=3, color="#3b7ea1")
axes[0].set_xticks(pos)
axes[0].set_xticklabels(table["model"], rotation=15)
axes[0].set_ylabel("MAE (lower is better)")
axes[0].set_title("Mean absolute error")
axes[1].bar(pos, table["r2"], yerr=table["r2_sd"], capsize=3, color="#e9a23b")
axes[1].axhline(0, color="grey", linewidth=1)
axes[1].set_xticks(pos)
axes[1].set_xticklabels(table["model"], rotation=15)
axes[1].set_ylabel("R-squared (higher is better)")
axes[1].set_title("R-squared")
fig.tight_layout()
fig.savefig(f"{args.out}/model_comparison.png", dpi=200)
plt.close(fig)

pred = oof_pred[best]
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(y, pred, s=45, color="#3b7ea1", edgecolor="white")
lim = [0, max(y.max(), pred.max()) + 0.05]
ax.plot(lim, lim, "k--", linewidth=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Actual tail fraction")
ax.set_ylabel("Predicted tail fraction")
ax.set_title(f"Predicted vs actual: {best}")
fig.tight_layout()
fig.savefig(f"{args.out}/predicted_vs_actual.png", dpi=200)
plt.close(fig)

resid = y - pred
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(pred, resid, s=40, color="#e76f51", edgecolor="white")
axes[0].axhline(0, color="black", linewidth=1)
axes[0].set_xlabel("Predicted tail fraction")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs predicted")
axes[1].hist(resid, bins=8, color="#2a9d8f")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual distribution")
fig.tight_layout()
fig.savefig(f"{args.out}/residuals.png", dpi=200)
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
ax.hist(y, bins=10, color="#8a6fb3", edgecolor="white")
ax.axvline(y.mean(), color="black", linestyle="--", label=f"mean {y.mean():.2f}")
ax.set_xlabel("Tail fraction")
ax.set_ylabel("Count")
ax.set_title("Distribution of the target")
ax.legend()
fig.tight_layout()
fig.savefig(f"{args.out}/target_distribution.png", dpi=200)
plt.close(fig)

write_provenance(args.out, provenance, extra={
    "analysis": "regression on tail_fraction",
    "folds": args.folds, "repeats": args.repeats, "seed": args.seed,
    "target_mean": float(y.mean()),
    "target_min": float(y.min()), "target_max": float(y.max()),
    "best_model_by_mae": best,
    "caveat": ("A negative held-out R2 at this corpus size means the features "
               "carry less signal than the target's own mean, which at small n "
               "is as much a sample-size statement as a modelling one. Read "
               "model_scores.csv with n_rows in mind."),
})

print(f"\nfigures and tables written to {args.out}/")
