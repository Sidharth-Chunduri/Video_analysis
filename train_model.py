import os
import json
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

PARAM_PATH = "test_features"
SCORE_PATH = "Video_Analysis_Results.csv"

def load_feature_jsons(folder_path):
    features = {}
    for fname in os.listdir(folder_path):
        if fname.endswith(".json"):
            video_id = os.path.splitext(fname)[0].replace("features_", "")
            with open(os.path.join(folder_path, fname), "r") as f:
                features[video_id] = json.load(f)
    return pd.DataFrame.from_dict(features, orient='index').sort_index()

def train():
    features_df = load_feature_jsons(PARAM_PATH) 
    features_df.index = features_df.index.astype(str)

    llm_df = pd.read_csv(SCORE_PATH, index_col=0)
    llm_df.index = llm_df.index.astype(str) 

    X = features_df.loc[llm_df.index]
    y = llm_df

    print(y.describe())

    # Fill missing values with column mean
    X_filled = X.fillna(X.mean())
    y = y.loc[X_filled.index]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_filled), columns=X_filled.columns, index=X_filled.index)

    models = {}
    for target in y.columns:
        print(f"Tuning for: {target}")
        model = HistGradientBoostingRegressor(random_state=42)
        grid = GridSearchCV(model, {
            'max_iter': [100, 300, 500],
            'max_depth': [4, 6, 8, None],
            'min_samples_leaf': [1, 3, 5]
        }, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_scaled, y[target])
        print(f"Best params for {target}: {grid.best_params_}")
        print(f"Best CV MSE: {-grid.best_score_:.2f}")
        models[target] = grid.best_estimator_
        joblib.dump(models[target], f"model_{target}.pkl")
    
    return models, X_scaled, y

models, X, y = train()

target_to_plot = "confidence"
result = permutation_importance(
    models[target_to_plot], X, y[target_to_plot], n_repeats=10, random_state=42, n_jobs=-1
)
importances = result.importances_mean
feat_names = X.columns

plt.figure(figsize=(10, 5))
plt.barh(feat_names, importances)
plt.xlabel("Permutation Importance")
plt.title(f"Top Features for '{target_to_plot}'")
plt.tight_layout()
plt.show()

for target in y.columns:
    y_true = y[target]
    y_pred = models[target].predict(X)
    plt.scatter(y_true, y_pred)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{target}: True vs Predicted")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.tight_layout()
    plt.show()

print(X.corrwith(y['confidence']))