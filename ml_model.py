import pandas as pd
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

#Load dataset
file_path = input("Enter path to CSV file: ")
df = pd.read_csv(file_path)

#Display columns
print("\nAvailable columns:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")
target_index = int(input("Enter the index of the target column: "))
target_col = df.columns[target_index]


target_col = df.columns[target_index]

# Drop rows missing target
df = df.dropna(subset=[target_col])


y_raw = df[target_col]

if pd.api.types.is_numeric_dtype(y_raw) and len(np.unique(y_raw)) > 2:
    df[target_col] = pd.cut(y_raw, bins=2, labels=["Low", "High"])
    y = LabelEncoder().fit_transform(df[target_col])
elif y_raw.dtype == 'object' and len(np.unique(y_raw)) > 2:
    main_class = y_raw.mode()[0]
    df[target_col] = y_raw.apply(lambda x: main_class if x == main_class else f"Not_{main_class}")
    y = LabelEncoder().fit_transform(df[target_col])
else:
    if y_raw.dtype == 'object':
        y = LabelEncoder().fit_transform(y_raw)
    else:
        y = y_raw

is_binary = len(np.unique(y)) == 2


# Encode f
X = df.drop(columns=[target_col])
X = pd.get_dummies(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}
roc_curves = {}
conf_matrices = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary' if is_binary else 'macro')

    cm = confusion_matrix(y_test, y_pred)
    conf_matrices[name] = cm  

    # ROC (for binary only)
    if is_binary:
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            roc_curves[name] = (fpr, tpr, roc_auc)
        except Exception as e:
            print(f"Skipping ROC for {name}: {e}")
    else:
        print(f"Skipping ROC for {name}: target is not binary.")

    results[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "training_time": round(training_time, 4),
        "confusion_matrix": cm.tolist()
    }
print("\nJSON-friendly evaluation output:")
print(json.dumps(results, indent=4))

best_precision_model = max(results.items(), key=lambda x: x[1]['precision'])
print(f"\nBest model by precision: {best_precision_model[0]} (Precision: {best_precision_model[1]['precision']})")
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\nBest model by accuracy: {best_model[0]} (Accuracy: {best_model[1]['accuracy']})")
if roc_curves:
    best_auc_model = max(roc_curves.items(), key=lambda x: x[1][2])
    print(f"Best model by AUC: {best_auc_model[0]} (AUC: {best_auc_model[1][2]:.4f})")

# Plot confusion matrices and ROC curve
num_models = len(conf_matrices)
has_roc = bool(roc_curves)
fig_cols = num_models + (1 if has_roc else 0)

plt.figure(figsize=(5 * fig_cols, 5))

# Plot confusion matrices
for i, (name, cm) in enumerate(conf_matrices.items()):
    plt.subplot(1, fig_cols, i + 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix\n{name}')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

# Plot ROC curves 
if has_roc:
    plt.subplot(1, fig_cols, fig_cols)
    for name, (fpr, tpr, roc_auc) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()