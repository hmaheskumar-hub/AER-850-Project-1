# AER850 – Project 1 
# Hajaanan Maheskumar - 501099977

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import joblib

def report_scores(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n[{name}] ACC={acc:.4f}  PREC_w={prec:.4f}  REC_w={rec:.4f}  F1_w={f1w:.4f}")
    return {"model": name, "acc": acc, "prec_w": prec, "rec_w": rec, "f1_w": f1w}

def plot_confusion_heatmap(y_true, y_pred, title="Confusion matrix", labels=None, vmax=None):
    cm = confusion_matrix(y_true, y_pred)
    n = cm.shape[0]
    if labels is None:
        labels = list(range(1, n+1))
    if vmax is None:
        vmax = cm.max()
    plt.figure(figsize=(6.2, 4.4), dpi=150)
    im = plt.imshow(cm, cmap="viridis", vmin=0, vmax=vmax)
    plt.colorbar(im)
    plt.xticks(np.arange(n), labels)
    plt.yticks(np.arange(n), labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            color = "white" if val > (0.6 * vmax) else "yellow"
            plt.text(j, i, str(val), ha="center", va="center", fontsize=8, color=color)
    plt.tight_layout()
    plt.show()

df = pd.read_csv("Project 1 Data.csv")
X = df[["X", "Y", "Z"]].copy()
y = df["Step"].astype(int)

print("Head:\n", df.head())
print("\nNulls:\n", df.isnull().sum())
print("\nClass share:\n", y.value_counts(normalize=True).sort_index())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
pts = ax.scatter(df["X"], df["Y"], df["Z"], c=df["Step"], cmap="cividis", s=20, alpha=0.9)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
fig.colorbar(pts, ax=ax, label="Step")
plt.title("3D scatter of coordinates by step")
plt.show()

corr = df[["X","Y","Z","Step"]].corr()
plt.figure(figsize=(6.4,5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.4)
plt.title("Correlation heatmap (X, Y, Z, Step)")
plt.tight_layout(); plt.show()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe_lr = Pipeline([("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=3000))])

pipe_svc = Pipeline([("scaler", StandardScaler()),
                     ("clf", SVC())])

pipe_rf = Pipeline([("scaler", StandardScaler(with_mean=False)),
                    ("clf", RandomForestClassifier(random_state=42))])

pipe_dt = Pipeline([("scaler", StandardScaler(with_mean=False)),
                    ("clf", DecisionTreeClassifier(random_state=42))])

grid_lr = {"clf__C": [0.05, 0.1, 0.5, 1.0, 2.0]}
rand_svc = {"clf__C": np.logspace(-2, 2, 21),
            "clf__kernel": ["rbf", "poly", "sigmoid"],
            "clf__gamma": ["scale", "auto"]}
grid_rf = {"clf__n_estimators": [120, 200, 320],
           "clf__max_depth": [None, 8, 12, 16],
           "clf__min_samples_split": [2, 4, 6],
           "clf__min_samples_leaf": [1, 2, 3]}
grid_dt = {"clf__max_depth": [None, 6, 10, 14, 18],
           "clf__min_samples_split": [2, 4, 6, 8],
           "clf__min_samples_leaf": [1, 2, 4]}

gs_lr  = GridSearchCV(pipe_lr, grid_lr, cv=cv, n_jobs=-1, scoring="f1_weighted")
rs_svc = RandomizedSearchCV(pipe_svc, rand_svc, n_iter=28, cv=cv, n_jobs=-1,
                            random_state=42, scoring="f1_weighted")
gs_rf  = GridSearchCV(pipe_rf, grid_rf, cv=cv, n_jobs=-1, scoring="f1_weighted")
gs_dt  = GridSearchCV(pipe_dt, grid_dt, cv=cv, n_jobs=-1, scoring="f1_weighted")

gs_lr.fit(X_train, y_train)
rs_svc.fit(X_train, y_train)
gs_rf.fit(X_train, y_train)
gs_dt.fit(X_train, y_train)

print("\nBest params:")
print("  LR:", gs_lr.best_params_)
print("  SVC:", rs_svc.best_params_)
print("  RF:", gs_rf.best_params_)
print("  DT:", gs_dt.best_params_)

models = {
    "Logistic Regression": gs_lr.best_estimator_,
    "SVC": rs_svc.best_estimator_,
    "Random Forest": gs_rf.best_estimator_,
    "Decision Tree": gs_dt.best_estimator_,
}

rows = []
preds = {}
global_vmax = 0
for name, est in models.items():
    yp = est.predict(X_test)
    preds[name] = yp
    rows.append(report_scores(y_test, yp, name))
    global_vmax = max(global_vmax, confusion_matrix(y_test, yp).max())

best = max(rows, key=lambda d: d["f1_w"])["model"]
print(f"\nBest single model (by weighted F1): {best}")

for name in models.keys():
    plot_confusion_heatmap(
        y_test, preds[name],
        title=f"{name}: Confusion matrix",
        labels=list(range(1, 14)),
        vmax=global_vmax
    )

print("\nClassification report (best):")
print(classification_report(y_test, preds[best], zero_division=0))

stack = StackingClassifier(
    estimators=[("rf", models["Random Forest"]), ("svc", models["SVC"])],
    final_estimator=LogisticRegression(max_iter=3000),
    cv=cv, n_jobs=-1
)
stack.fit(X_train, y_train)
yps = stack.predict(X_test)
rows.append(report_scores(y_test, yps, "Stacked (RF+SVC→LR)"))
global_vmax = max(global_vmax, confusion_matrix(y_test, yps).max())
plot_confusion_heatmap(
    y_test, yps,
    title="Stacked (RF+SVC→LR): Confusion matrix",
    labels=list(range(1, 14)),
    vmax=global_vmax
)

pd.DataFrame(rows).to_csv("model_metrics.csv", index=False)

joblib.dump(models[best], "model_best_single.joblib")
joblib.dump(stack, "model_stacked.joblib")
print("\nSaved models: model_best_single.joblib, model_stacked.joblib")

new_coords = pd.DataFrame([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0.0000, 3.0625, 1.93],
    [9.4000, 3.0000, 1.80],
    [9.4000, 3.0000, 1.30],
], columns=["X","Y","Z"])

pred_new = stack.predict(new_coords)
print("\nPredictions (stacked) for new coordinates:", pred_new)
