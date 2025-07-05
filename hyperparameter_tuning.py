import utils
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run_hyperparameter_tuning():
    df = utils.load_data()
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Ridge Regression": (Ridge(), {
            "alpha": [0.1, 1.0, 10.0],
            "solver": ['auto', 'svd', 'cholesky']
        }),
        "Decision Tree": (DecisionTreeRegressor(), {
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }),
        "Random Forest": (RandomForestRegressor(), {
            "n_estimators": [50, 100],
            "max_depth": [5, 10],
            "min_samples_split": [2, 5]
        })
    }

    for name, (model, params) in models.items():
        print(f"\n{name} - Hyperparameter Tuning")
        grid = GridSearchCV(model, params, cv=3, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)

        print("Best Params:", grid.best_params_)
        print("Best CV R2 Score:", grid.best_score_)

        y_pred = grid.predict(X_test)
        print("Test MSE:", mean_squared_error(y_test, y_pred))
        print("Test R2 Score:", r2_score(y_test, y_pred))
        print("-" * 40)

if __name__ == "__main__":
    run_hyperparameter_tuning()

