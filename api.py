from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

# Autoriser les requêtes Cross-Origin (depuis ton Streamlit local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou l'url de ton client Streamlit en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = "random_forest"):
    try:
        # Lecture du CSV uploadé
        content = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(content))
        
        # Ici tu lanceras ton modèle ML selon model_name sur df
        # Pour l’exemple on renvoie un faux résultat
        results = []
        for idx in range(len(df)):
            results.append({"index": idx, "prediction": 0, "probability_faux": 0.1})
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
