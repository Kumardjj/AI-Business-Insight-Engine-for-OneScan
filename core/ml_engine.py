import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

class InventoryMLEngine:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.data_path = "data/mock_inventory_data.csv"
        
    def generate_mock_data(self, num_products=50):
        """Generates a realistic retail inventory dataset."""
        os.makedirs("data", exist_ok=True)
        np.random.seed(42)
        
        categories = ['Electronics', 'Apparel', 'Home & Kitchen', 'Beauty', 'Groceries']
        data = []
        
        for i in range(1, num_products + 1):
            product_id = f"PROD_{1000 + i}"
            category = np.random.choice(categories)
            current_stock = np.random.randint(5, 500)
            avg_daily_sales = np.random.uniform(5, 40)
            lead_time_days = np.random.randint(2, 15)  # Days to restock
            supplier_reliability = np.random.uniform(0.7, 1.0)
            
            # Feature calculation
            days_to_depletion = current_stock / avg_daily_sales if avg_daily_sales > 0 else 999
            safety_stock = avg_daily_sales * (lead_time_days * 1.2)
            
            # Rule-based target label generation for training
            if days_to_depletion <= lead_time_days:
                urgency = "High"
            elif days_to_depletion <= (lead_time_days * 2):
                urgency = "Medium"
            else:
                urgency = "Low"
                
            data.append([
                product_id, category, current_stock, round(avg_daily_sales, 2), 
                lead_time_days, round(supplier_reliability, 2), round(safety_stock, 2), urgency
            ])
            
        columns = [
            'product_id', 'category', 'current_stock', 'avg_daily_sales', 
            'lead_time_days', 'supplier_reliability', 'safety_stock', 'restock_urgency'
        ]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(self.data_path, index=False)
        return df

    def train_model(self):
        """Loads data, engineers features, and trains the Random Forest classifier."""
        if not os.path.exists(self.data_path):
            df = self.generate_mock_data()
        else:
            df = pd.read_csv(self.data_path)
            
        # Features & Target
        X = df[['current_stock', 'avg_daily_sales', 'lead_time_days', 'supplier_reliability', 'safety_stock']]
        y = df['restock_urgency']
        
        # Encode target labels (High=0, Low=1, Medium=2 etc. depending on fit)
        y_encoded = self.label_encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        return "Model trained successfully."

    def predict_urgency(self, current_stock, avg_daily_sales, lead_time_days, supplier_reliability):
        """Predicts the stockout urgency risk for a given item state."""
        safety_stock = avg_daily_sales * (lead_time_days * 1.2)
        features = np.array([[current_stock, avg_daily_sales, lead_time_days, supplier_reliability, safety_stock]])
        
        pred_encoded = self.model.predict(features)
        pred_label = self.label_encoder.inverse_transform(pred_encoded)[0]
        
        # Calculate calculated metrics to pass to LLM context
        days_to_depletion = current_stock / avg_daily_sales if avg_daily_sales > 0 else 999
        
        return {
            "predicted_urgency": pred_label,
            "safety_stock_threshold": round(safety_stock, 2),
            "estimated_days_left": round(days_to_depletion, 1)
        }