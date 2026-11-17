import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def compare_models(df, random_state=42):
    X, y = df[FEATURES], df["label"]
    counts = y.value_counts()
    if counts.min() < 3:
        raise ValueError("Each crop class needs at least 3 rows for a train/test split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
        "KNN": Pipeline([("imputer", SimpleImputer()), ("scale", StandardScaler()),
                         ("model", KNeighborsClassifier(n_neighbors=5))]),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=random_state,
                                                 class_weight="balanced", min_samples_leaf=2)
    }
    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
        rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                     "Macro precision": p, "Macro recall": r, "Macro F1": f})
        fitted[name] = model
    best_name = max(rows, key=lambda z: z["Macro F1"])["Model"]
    best = fitted[best_name]
    return best, best_name, pd.DataFrame(rows), X_test, y_test, best.predict(X_test)

def feature_importance(model):
    core = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if hasattr(core, "feature_importances_"):
        return pd.DataFrame({"Feature": FEATURES, "Importance": core.feature_importances_}).sort_values("Importance", ascending=False)
    return pd.DataFrame({"Feature": FEATURES, "Importance": np.zeros(len(FEATURES))})
