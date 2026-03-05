import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def train_model():

    # Load dataset
    data = pd.read_csv("dataset/weather.csv")

    # Features
    X = data[['Rel Hum_%', 'Wind Speed_km/h', 'Press_kPa']]

    # Target
    y = data['Temp_C']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestRegressor()

    # Train
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "weather_model.pkl")


# Run training if file executed directly
if __name__ == "__main__":
    train_model()