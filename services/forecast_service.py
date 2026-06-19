import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from app.config import MODEL_DIR, REPORTS_DIR, MAX_SAMPLE_SIZE

logger = logging.getLogger("ParkTwinAI.ForecastService")

class ForecastService:
    def __init__(self, df=None):
        self.df = df
        self.model = None
        self.station_encoder = LabelEncoder()
        self.vehicle_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_names = ['hour', 'day_of_week', 'month', 'police_station_enc', 'vehicle_type_enc', 'historical_frequency']

    def prepare_data(self, df=None):
        """Prepare aggregations and features for XGBoost model training."""
        if df is not None:
            self.df = df
            
        if self.df is None or len(self.df) == 0:
            raise ValueError("No dataframe available for model preparation.")
            
        df_copy = self.df.copy()
        
        # Extract date-hour fields
        df_copy['date'] = df_copy['created_datetime'].dt.date
        df_copy['hour'] = df_copy['created_datetime'].dt.hour
        df_copy['day_of_week'] = df_copy['created_datetime'].dt.dayofweek
        df_copy['month'] = df_copy['created_datetime'].dt.month
        
        logger.info("Aggregating data by spatial-temporal bins...")
        # Group by date, hour, police_station, vehicle_type to find counts of violations
        agg_df = df_copy.groupby(['date', 'hour', 'day_of_week', 'month', 'police_station', 'vehicle_type']).size().reset_index(name='violation_count')
        
        # Calculate historical frequency: average violations for this (police_station, hour)
        hist_freq = df_copy.groupby(['police_station', 'hour']).size().reset_index(name='total_hist')
        # Total number of unique dates in historical data
        num_dates = df_copy['date'].nunique()
        if num_dates == 0:
            num_dates = 1
        hist_freq['historical_frequency'] = hist_freq['total_hist'] / num_dates
        
        # Merge back to aggregated data
        agg_df = pd.merge(agg_df, hist_freq[['police_station', 'hour', 'historical_frequency']], on=['police_station', 'hour'], how='left')
        agg_df['historical_frequency'] = agg_df['historical_frequency'].fillna(0.0)
        
        # Encode categorical fields
        self.station_encoder.fit(df_copy['police_station'].unique().tolist() + ["Unknown Station"])
        self.vehicle_encoder.fit(df_copy['vehicle_type'].unique().tolist() + ["UNKNOWN"])
        
        agg_df['police_station_enc'] = self.station_encoder.transform(agg_df['police_station'])
        agg_df['vehicle_type_enc'] = self.vehicle_encoder.transform(agg_df['vehicle_type'])
        
        # Define binary target: 1 if violation_count > median count (indicating higher-than-average risk), else 0
        median_val = agg_df['violation_count'].median()
        agg_df['high_risk'] = (agg_df['violation_count'] > median_val).astype(int)
        
        return agg_df

    def train_model(self, df=None):
        """Train XGBoost model and save evaluation metrics."""
        agg_df = self.prepare_data(df)
        
        if len(agg_df) < 50:
            # Not enough data, create synthetic rows for training robustness
            logger.warning("Very small aggregated dataset. Synthesizing records for training...")
            synthetic_rows = []
            for i in range(100):
                synthetic_rows.append({
                    'hour': i % 24,
                    'day_of_week': i % 7,
                    'month': (i % 12) + 1,
                    'police_station_enc': i % 5,
                    'vehicle_type_enc': i % 4,
                    'historical_frequency': float(np.random.randint(0, 10)),
                    'high_risk': np.random.choice([0, 1])
                })
            train_df = pd.DataFrame(synthetic_rows)
        else:
            # Sample for performance if needed
            if len(agg_df) > MAX_SAMPLE_SIZE:
                train_df = agg_df.sample(n=MAX_SAMPLE_SIZE, random_state=42)
            else:
                train_df = agg_df
                
        X = train_df[self.feature_names]
        y = train_df['high_risk']
        
        # Handle single class edge cases
        if len(y.unique()) < 2:
            logger.warning("Target has only one class. Simulating class balance...")
            # Toggle some labels to avoid training failures
            y.iloc[::2] = 1 - y.iloc[::2]
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logger.info("Training XGBoost Classifier...")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Predict & Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5)
        }
        
        logger.info(f"Model trained successfully. Accuracy: {metrics['accuracy']:.4f}")
        
        # Persist model
        model_path = MODEL_DIR / "xgb_hotspot_model.json"
        self.model.save_model(str(model_path))
        
        # Save evaluation metrics
        with open(REPORTS_DIR / "model_evaluation.json", 'w') as f:
            json.dump(metrics, f, indent=4)
            
        return metrics

    def predict_tomorrow_hotspots(self, df=None):
        """Forecast tomorrow's risk probabilities for all police stations and hours."""
        if not self.is_trained:
            self.train_model(df)
            
        tomorrow = datetime.now() + timedelta(days=1)
        t_day_of_week = tomorrow.weekday()
        t_month = tomorrow.month
        
        # Get list of unique stations and vehicle types
        stations = self.station_encoder.classes_
        vehicles = self.vehicle_encoder.classes_
        
        # Precompute historical frequencies
        df_copy = self.df.copy()
        df_copy['date'] = df_copy['created_datetime'].dt.date
        df_copy['hour'] = df_copy['created_datetime'].dt.hour
        hist_freq = df_copy.groupby(['police_station', 'hour']).size().reset_index(name='total_hist')
        num_dates = df_copy['date'].nunique() or 1
        hist_freq['historical_frequency'] = hist_freq['total_hist'] / num_dates
        
        predict_records = []
        # Construct combinations for the 24 hours of tomorrow for major stations/vehicles
        # To avoid massive Cartesian product, select top 10 stations and top 3 vehicle types
        top_stations = list(df_copy['police_station'].value_counts().head(10).index)
        if "Unknown Station" not in top_stations:
            top_stations.append("Unknown Station")
            
        top_vehicles = list(df_copy['vehicle_type'].value_counts().head(3).index)
        if "UNKNOWN" not in top_vehicles:
            top_vehicles.append("UNKNOWN")
            
        for station in top_stations:
            for vehicle in top_vehicles:
                for hour in range(24):
                    # Find historical frequency
                    freq_row = hist_freq[(hist_freq['police_station'] == station) & (hist_freq['hour'] == hour)]
                    h_freq = float(freq_row['historical_frequency'].values[0]) if not freq_row.empty else 0.0
                    
                    predict_records.append({
                        "station": station,
                        "vehicle_type": vehicle,
                        "hour": hour,
                        "day_of_week": t_day_of_week,
                        "month": t_month,
                        "police_station_enc": self.station_encoder.transform([station])[0],
                        "vehicle_type_enc": self.vehicle_encoder.transform([vehicle])[0],
                        "historical_frequency": h_freq
                    })
                    
        predict_df = pd.DataFrame(predict_records)
        X_predict = predict_df[self.feature_names]
        
        # Predict risk probabilities
        probs = self.model.predict_proba(X_predict)[:, 1]
        predict_df['risk_probability'] = probs
        
        # Sort and get top 10 predicted hotspots
        top_10 = predict_df.sort_values(by="risk_probability", ascending=False).head(10).reset_index(drop=True)
        
        # Format output
        results = []
        for i, row in top_10.iterrows():
            results.append({
                "rank": i + 1,
                "police_station": row['station'],
                "vehicle_type": row['vehicle_type'],
                "hour_of_day": int(row['hour']),
                "risk_probability": float(row['risk_probability']),
                "historical_frequency": float(row['historical_frequency'])
            })
            
        # Save forecast to reports
        with open(REPORTS_DIR / "tomorrow_forecast.json", 'w') as f:
            json.dump(results, f, indent=4)
            
        return results

    def get_feature_importance(self):
        """Return feature importances for features used in training."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained yet.")
        importances = self.model.feature_importances_
        feat_imp = {self.feature_names[i]: float(importances[i]) for i in range(len(self.feature_names))}
        # Sort by importance
        feat_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
        return feat_imp

if __name__ == "__main__":
    from app.data_pipeline import DataPipeline
    dp = DataPipeline()
    df = dp.load_data()
    fs = ForecastService(df)
    metrics = fs.train_model()
    print("Metrics:", metrics)
    top_10 = fs.predict_tomorrow_hotspots()
    print("Top Predicted Hotspots Tomorrow:")
    print(top_10[:3])
