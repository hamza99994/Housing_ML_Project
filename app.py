
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel


# 1. Create our FastAPI application
app = FastAPI(title="Housing ML App")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"


# 2. List our saved models
model_files = {
    "linear_regression": "linear_regression.joblib",
    "logistic_regression": "logistic_regression.joblib",
    "decision_tree_gini": "decision_tree_gini.joblib",
    "decision_tree_entropy": "decision_tree_entropy.joblib",
    "regression_tree": "regression_tree.joblib",
    "random_forest_classifier": "random_forest_classifier.joblib",
    "random_forest_regressor": "random_forest_regressor.joblib",
    "svm_classifier": "svm_classifier.joblib",
    "svr_regressor": "svr_regressor.joblib",
    "naive_bayes": "naive_bayes.joblib"
}


# 3. Load the trained models
models = {}

for name, filename in model_files.items():
    path = MODEL_DIR / filename

    if path.exists():
        models[name] = joblib.load(path)
    else:
        print(f"Missing model: {filename}")


# 4. Define house input fields
class HouseInput(BaseModel):
    price: float | None = None
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    parking: int
    prefarea: str
    furnishingstatus: str | None = None


# 5. Identify classification models
classification_models = {
    "logistic_regression",
    "decision_tree_gini",
    "decision_tree_entropy",
    "random_forest_classifier",
    "svm_classifier",
    "naive_bayes"
}


# 6. List available models
@app.get("/models")
def get_models():
    return {
        "available_models": list(models.keys())
    }


# 7. Make a prediction
@app.post("/predict/{model_name}")
def predict(model_name: str, house: HouseInput):

    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail="Model not found or not loaded"
        )

    data = house.model_dump()

    if model_name in classification_models:

        if data["price"] is None:
            raise HTTPException(
                status_code=400,
                detail="Price is required for classification"
            )

        data.pop("furnishingstatus")

    else:

        if data["furnishingstatus"] is None:
            raise HTTPException(
                status_code=400,
                detail="Furnishing status is required for regression"
            )

        data.pop("price")

    input_df = pd.DataFrame([data])

    prediction = models[model_name].predict(input_df)[0]

    if model_name in classification_models:
        return {
            "model": model_name,
            "prediction_type": "classification",
            "prediction": str(prediction)
        }

    return {
        "model": model_name,
        "prediction_type": "regression",
        "prediction": round(float(prediction), 2)
    }


# 8. Serve the frontend
STATIC_DIR = BASE_DIR / "static"

app.mount(
    "/css",
    StaticFiles(directory=STATIC_DIR / "css"),
    name="css"
)


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "html" / "index.html")
