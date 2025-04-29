from io import BytesIO

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

# Load model from MLflow
logged_model = "../mlartifacts/594638292518005964/62f5b07adca443d28078e22eeda10017/artifacts/model"
try:
    model = mlflow.pyfunc.load_model(logged_model)
except Exception as e:
    raise RuntimeError(f"Failed to load model from MLflow: {e}")


file_param = File(...)


@app.post("/predict/")
async def predict(file: UploadFile = file_param):
    try:
        # Read the uploaded file into a pandas DataFrame
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents), parse_dates=["Timestamp"])

        # Preserve timestamp as index
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df.set_index("Timestamp", inplace=True)

        # Settings
        target_column = "CO_reference_mg_per_m3"
        time_columns = ["Hour", "DayOfWeek", "Month"]
        to_drop = [target_column] + time_columns

        # Load scaler and encoder
        scaler = joblib.load("../models/scaler.pkl")
        encoder = joblib.load("../models/encoder.pkl")

        # Standardize numerical columns
        cols_to_scale = [c for c in df.columns if c not in to_drop]
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

        # One-hot encode time-related columns
        encoded_time = encoder.transform(df[time_columns])
        encoded_time_df = pd.DataFrame(
            encoded_time, columns=encoder.get_feature_names_out(time_columns), index=df.index
        )

        # Drop unnecessary columns and prepare final DataFrame
        df = df.drop(columns=to_drop).join(encoded_time_df)

        # Reorder final feature matrix to match training
        numeric_feats = scaler.feature_names_in_.tolist()
        time_feats = encoder.get_feature_names_out(time_columns).tolist()
        df = df[numeric_feats + time_feats]

        # Perform predictions
        predictions = model.predict(df)

        # Return predictions as JSON
        results = []
        for ts, pred in zip(df.index, predictions):
            results.append({"timestamp": ts.isoformat(), "co_mg_per_m3": float(pred)})

        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
