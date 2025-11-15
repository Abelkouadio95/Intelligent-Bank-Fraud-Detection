from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(
    title="FraudeShield - Détection de Fraude Bancaire",
    description="Système de détection de fraude bancaire utilisant l'apprentissage automatique",
)

templates = Jinja2Templates(directory="app/templates")

# Chargement du modèle
MODEL_PATH = Path(__file__).parent / "models" / "Modelknn.pkl"

try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"Modèle chargé avec succès depuis {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Modèle non trouvé à {MODEL_PATH}")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    model = None    


# Modèle Pydantic pour la validation des données d'entrée
class TransactionRequest(BaseModel):
    type: int = Field(..., description="Type de transaction: 0 pour CASH_OUT, 1 pour TRANSFER", ge=0, le=1)
    amount: float = Field(..., description="Montant de la transaction", ge=0)
    oldbalanceOrg: float = Field(..., description="Ancien solde du compte d'origine", ge=0)

# Page prncipale
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_fraud(transaction: TransactionRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas disponible."
        )
    
    try:
        # Préparation des données pour la prédiction
        features = np.array([[transaction.type, transaction.amount, transaction.oldbalanceOrg]])
        
        # Prédiction
        prediction = model.predict(features)[0]
        
        # Probabilité 
        try:
            probabilities = model.predict_proba(features)[0]
            fraud_probability = float(probabilities[1]) if len(probabilities) > 1 else 0.0
        except:
            fraud_probability = None
        
        return JSONResponse({
            "is_fraud": bool(prediction),
            "fraud_probability": fraud_probability,
            "transaction_type": "TRANSFER" if transaction.type == 1 else "CASH_OUT",
            "amount": transaction.amount,
            "oldbalanceOrg": transaction.oldbalanceOrg,
            "status": "fraud" if prediction == 1 else "legitimate"
        })
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )
