import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class InventoryInsight(BaseModel):
    alert_level: str = Field(description="CRITICAL, WARNING, or STABLE based on urgency.")
    human_readable_summary: str = Field(description="A concise 1-2 sentence executive summary.")
    action_items: list[str] = Field(description="Actionable steps for the retail manager.")
    recommended_reorder_quantity: int = Field(description="Optimal units to order.")

class GenAIAssistant:
    def __init__(self):
        self.client = genai.Client()
        # Fallback structure targeting robust configurations
        self.primary_model = "gemini-2.5-flash"
        self.fallback_model = "gemini-2.5-pro"

    def generate_smart_recommendation(self, product_id: str, category: str, ml_results: dict, current_stock: int) -> InventoryInsight:
        prompt = f"""
        Analyze this inventory state:
        Product ID: {product_id} ({category})
        Current Stock: {current_stock} units
        ML Urgency Prediction: {ml_results['predicted_urgency']}
        Safety Stock Limit: {ml_results['safety_stock_threshold']} units
        Days Until Stockout: {ml_results['estimated_days_left']} days
        """

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=InventoryInsight,
            temperature=0.2,
        )

        try:
            print(f"🤖 Sending data payload to primary model: {self.primary_model}...")
            response = self.client.models.generate_content(
                model=self.primary_model,
                contents=prompt,
                config=config,
            )
            return response.parsed

        except Exception as e:
            print(f"⚠️ Primary model engine ran into an error: {str(e)}")
            print(f"🔄 Activating fail-safe router: Switching to {self.fallback_model}...")
            
            try:
                response = self.client.models.generate_content(
                    model=self.fallback_model,
                    contents=prompt,
                    config=config,
                )
                print("✅ Fail-safe routing succeeded. Returning structural data.")
                return response.parsed
            except Exception as fallback_error:
                # Direct, actionable logging
                error_msg = f"Both Google Gemini endpoints failed. Primary Error: {str(e)} | Fallback Error: {str(fallback_error)}"
                print(f"❌ Critical Core Failure: {error_msg}")
                
                # Create a mock Pydantic object containing the error details 
                # This prevents FastAPI from throwing a 500 and forces it to display the real issue!
                return InventoryInsight(
                    alert_level="ERROR",
                    human_readable_summary=f"System failed to reach Gemini API providers.",
                    action_items=[f"API Error Details: {str(fallback_error)}", "Check if GEMINI_API_KEY is properly loaded in your .env file."],
                    recommended_reorder_quantity=0
                )
            