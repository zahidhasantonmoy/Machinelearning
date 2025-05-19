import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso, HuberRegressor, PassiveAggressiveRegressor, ARDRegression, BayesianRidge, SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import joblib
import os

# Load data
df = pd.read_csv('Gold Price.csv')  # Update path if needed

# Prepare features and target
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.joblib')

# Define models
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'LinearRegression': LinearRegression(),
    'ElasticNet': ElasticNet(random_state=42),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'Huber': HuberRegressor(),
    'PassiveAggressive': PassiveAggressiveRegressor(random_state=42),
    'ARD': ARDRegression(),
    'BayesianRidge': BayesianRidge(),
    'SGD': SGDRegressor(random_state=42),
    'SVR': SVR(),
    'KNeighbors': KNeighborsRegressor(),
    'GaussianProcess': GaussianProcessRegressor(random_state=42)
}

# Train and save models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f'models/{name}.joblib')
    print(f"Saved {name} model.")