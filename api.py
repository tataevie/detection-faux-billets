from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from fastapi import Form
import io

app = FastAPI()

# Autoriser les requêtes Cross-Origin (depuis ton Streamlit local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou l'url de ton client Streamlit en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def test():
    return {"Greeting": "Hello World"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("random_forest")  # <-- Form ici
):
    print("success")
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        results = []
        for idx in range(len(df)):
            results.append({"index": idx, "prediction": 0, "probability_faux": 0.1})
        
        return {"results": results}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
