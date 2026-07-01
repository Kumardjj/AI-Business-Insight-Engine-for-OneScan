from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.ml_engine import InventoryMLEngine
from core.genai_engine import GenAIAssistant
import json
from contextlib import asynccontextmanager


app = FastAPI(title="AI-Powered Smart Inventory Assistant",
    description="Production-grade ML + GenAI system for retail stockout prediction and automated analytics.",
    version="1.0.0")

# Initialize engines on server boot
ml_engine = InventoryMLEngine()
genai_engine = GenAIAssistant()


# 2. Train the model immediately right here on startup execution
print("Initializing and training Random Forest Classifier...")
ml_engine.train_model()
print("Model trained successfully and ready for inference.")

# Ensure model is trained on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

# Request Validation Models
class InventoryAnalysisRequest(BaseModel):
    product_id: str
    category: str
    current_stock: int
    avg_daily_sales: float
    lead_time_days: int
    supplier_reliability: float


@app.get("/health")
def health_check():
    return {"status": "healthy", "engine": "FastAPI + Gemini 3.5"}

@app.post("/api/v1/analyze-stock")
async def analyze_stock_endpoint(payload: InventoryAnalysisRequest):
    try:
        # Step 1: Execute Predictive ML Analytics
        ml_results = ml_engine.predict_urgency(
            current_stock=payload.current_stock,
            avg_daily_sales=payload.avg_daily_sales,
            lead_time_days=payload.lead_time_days,
            supplier_reliability=payload.supplier_reliability
        )
        
        # Step 2: Inject ML Insights into cognitive GenAI Layer
        genai_json_str = genai_engine.generate_smart_recommendation(
            product_id=payload.product_id,
            category=payload.category,
            ml_results=ml_results,
            current_stock=payload.current_stock
        )
        
        # Parse the structured JSON response from Gemini
        ai_insights = json.loads(genai_json_str)
        
        # Step 3: Return the unified corporate data payload
        return {
            "product_id": payload.product_id,
            "category": payload.category,
            "predictive_metrics": ml_results,
            "ai_decision_support": ai_insights
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal System Error: {str(e)}")