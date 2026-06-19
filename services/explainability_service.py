import os
import logging
import matplotlib
matplotlib.use('Agg')  # Set backend to avoid window GUI issues in background threads
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from app.config import EXPLANATIONS_DIR

logger = logging.getLogger("ParkTwinAI.ExplainabilityService")

class ExplainabilityService:
    def __init__(self, forecast_service=None):
        self.fs = forecast_service

    def generate_explanations(self, df=None):
        """Generate SHAP values and save visual explanation plots."""
        if self.fs is None:
            raise ValueError("ForecastService is required to generate explanations.")
            
        if not self.fs.is_trained:
            logger.info("Training XGBoost model for explainability...")
            self.fs.train_model(df)
            
        model = self.fs.model
        feature_names = self.fs.feature_names
        
        # Prepare sample data for SHAP
        agg_df = self.fs.prepare_data(df)
        X = agg_df[feature_names]
        
        # Cap data size for explanation to keep it fast
        X_sample = X.sample(n=min(len(X), 200), random_state=42)
        
        try:
            import shap
            logger.info("Computing SHAP values using TreeExplainer...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Note: For binary classification, xgboost/shap sometimes returns a list of arrays 
            # (one for each class) or a single array. Let's handle both.
            if isinstance(shap_values, list):
                shap_val_array = shap_values[1] # Use positive class
            else:
                shap_val_array = shap_values
                
            # Create Explanation object if needed for waterfall plot
            explanation = shap.Explanation(
                values=shap_val_array,
                base_values=explainer.expected_value,
                data=X_sample.values,
                feature_names=feature_names
            )
            
            # Plot 1: SHAP Summary Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_val_array, X_sample, show=False)
            plt.title("SHAP Summary Plot (Feature Impact on Risk)", fontsize=14, pad=15)
            plt.tight_layout()
            summary_path = EXPLANATIONS_DIR / "shap_summary_plot.png"
            plt.savefig(summary_path, dpi=150)
            plt.close()
            logger.info(f"Saved SHAP summary plot to {summary_path}")
            
            # Plot 2: SHAP Feature Importance Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_val_array, X_sample, plot_type="bar", show=False)
            plt.title("SHAP Feature Importance (Average Magnitude)", fontsize=14, pad=15)
            plt.tight_layout()
            importance_path = EXPLANATIONS_DIR / "shap_feature_importance.png"
            plt.savefig(importance_path, dpi=150)
            plt.close()
            logger.info(f"Saved SHAP feature importance plot to {importance_path}")
            
            # Plot 3: SHAP Waterfall Plot for a sample high risk record
            # Find a high risk record (where prediction is high)
            preds = model.predict(X_sample)
            high_risk_indices = np.where(preds == 1)[0]
            idx = high_risk_indices[0] if len(high_risk_indices) > 0 else 0
            
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(explanation[idx], show=False)
            plt.title("SHAP Waterfall Plot for Single Hotspot Prediction", fontsize=14, pad=15)
            plt.tight_layout()
            waterfall_path = EXPLANATIONS_DIR / "shap_waterfall_plot.png"
            plt.savefig(waterfall_path, dpi=150)
            plt.close()
            logger.info(f"Saved SHAP waterfall plot to {waterfall_path}")
            
            return {
                "success": True,
                "method": "SHAP TreeExplainer",
                "summary_plot": str(summary_path),
                "importance_plot": str(importance_path),
                "waterfall_plot": str(waterfall_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to run SHAP explainability: {e}. Generating fallback explanations...")
            return self._generate_fallback_plots(model, X_sample, feature_names)

    def _generate_fallback_plots(self, model, X_sample, feature_names):
        """Fallback when SHAP fails, generating standard feature importance and attribution plots."""
        # 1. Feature Importance (from XGBoost directly)
        importances = model.feature_importances_
        indices = np.argsort(importances)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_names)), importances[indices], align='center', color='#1E88E5')
        plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title('Feature Importances (Global Attribution)')
        plt.tight_layout()
        importance_path = EXPLANATIONS_DIR / "shap_feature_importance.png"
        plt.savefig(importance_path, dpi=150)
        plt.close()
        
        # 2. Simulated Summary Plot (Correlation with Target)
        # Create a mock SHAP summary representation
        correlations = []
        for feat in feature_names:
            corr = float(np.corrcoef(X_sample[feat], model.predict(X_sample))[0, 1])
            correlations.append(corr if not np.isnan(corr) else 0.0)
            
        correlations = np.array(correlations)
        indices_corr = np.argsort(np.abs(correlations))
        
        plt.figure(figsize=(10, 6))
        colors = ['#EF5350' if c > 0 else '#26A69A' for c in correlations[indices_corr]]
        plt.barh(range(len(feature_names)), correlations[indices_corr], align='center', color=colors)
        plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices_corr])
        plt.xlabel('Feature correlation with Risk Score')
        plt.title('Feature Impact Summary (Operational Drivers)')
        plt.tight_layout()
        summary_path = EXPLANATIONS_DIR / "shap_summary_plot.png"
        plt.savefig(summary_path, dpi=150)
        plt.close()
        
        # 3. Simulated Waterfall Plot for high risk item
        preds = model.predict(X_sample)
        high_risk_indices = np.where(preds == 1)[0]
        idx = high_risk_indices[0] if len(high_risk_indices) > 0 else 0
        sample_row = X_sample.iloc[idx]
        
        # Attribute delays based on feature values relative to mean
        attributions = []
        means = X_sample.mean()
        for feat in feature_names:
            # Simple linear attribution proxy
            diff = sample_row[feat] - means[feat]
            val = diff * float(np.corrcoef(X_sample[feat], model.predict(X_sample))[0, 1])
            attributions.append(val if not np.isnan(val) else 0.0)
            
        attributions = np.array(attributions)
        
        plt.figure(figsize=(10, 6))
        # Draw waterfall steps
        cumulative = 0.5 # Base value
        for i, (feat, attr) in enumerate(zip(feature_names, attributions)):
            color = '#EF5350' if attr > 0 else '#26A69A'
            plt.bar(i, attr, bottom=cumulative, color=color, label=feat)
            cumulative += attr
            
        plt.xticks(range(len(feature_names)), feature_names, rotation=45)
        plt.ylabel('Attribution Contribution')
        plt.title('Waterfall Attribution Analysis (Decision Factors)')
        plt.tight_layout()
        waterfall_path = EXPLANATIONS_DIR / "shap_waterfall_plot.png"
        plt.savefig(waterfall_path, dpi=150)
        plt.close()
        
        return {
            "success": True,
            "method": "Attribution Logic Plots",
            "summary_plot": str(summary_path),
            "importance_plot": str(importance_path),
            "waterfall_plot": str(waterfall_path)
        }

if __name__ == "__main__":
    from app.data_pipeline import DataPipeline
    from services.forecast_service import ForecastService
    
    dp = DataPipeline()
    df = dp.load_data()
    fs = ForecastService(df)
    es = ExplainabilityService(fs)
    res = es.generate_explanations(df)
    print("Explanation results:", res)
