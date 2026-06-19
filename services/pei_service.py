import logging
import pandas as pd
import numpy as np
from app.config import PEI_WEIGHTS, REPORTS_DIR

logger = logging.getLogger("ParkTwinAI.PEIService")

class PEIService:
    def __init__(self, hotspots_summary_df=None):
        self.hotspots_df = hotspots_summary_df
        self.leaderboard = None

    def calculate_pei(self, hotspots_df=None):
        """Calculate the Parking Externality Index for each hotspot cluster."""
        if hotspots_df is not None:
            self.hotspots_df = hotspots_df
            
        if self.hotspots_df is None or self.hotspots_df.empty:
            logger.warning("Empty hotspots summary. Cannot calculate PEI.")
            return pd.DataFrame()
            
        df = self.hotspots_df.copy()
        
        # 1. Frequency Score (Normalized to 0-100)
        max_violations = df['violation_count'].max()
        df['freq_score'] = (df['violation_count'] / max_violations * 100.0) if max_violations > 0 else 0.0
        
        # 2. Duration Score (Average duration capped at 240 mins, normalized to 0-100)
        df['duration_score'] = df['avg_duration_minutes'].apply(lambda x: min((x / 240.0) * 100.0, 100.0))
        
        # 3. Peak Hour Severity Score (Peak ratio already 0-1, scale to 0-100)
        df['peak_score'] = df['peak_ratio'] * 100.0
        
        # 4. Junction Criticality Score (100 if named junction, 20 if No Junction)
        df['junction_score'] = df['primary_junction'].apply(lambda x: 20.0 if pd.isna(x) or str(x).lower() == "no junction" else 100.0)
        
        # 5. Density Score (Scale violation count in cluster as representation of spatial density)
        # Using a log scale for density to represent traffic bottlenecks better
        df['density_score'] = (np.log1p(df['violation_count']) / np.log1p(max_violations) * 100.0) if max_violations > 0 else 0.0
        
        # Calculate Weighted PEI
        df['pei_score'] = (
            PEI_WEIGHTS["frequency"] * df['freq_score'] +
            PEI_WEIGHTS["duration"] * df['duration_score'] +
            PEI_WEIGHTS["peak_hour"] * df['peak_score'] +
            PEI_WEIGHTS["junction_criticality"] * df['junction_score'] +
            PEI_WEIGHTS["density"] * df['density_score']
        )
        
        # Clean PEI score to 0-100
        df['pei_score'] = df['pei_score'].clip(0, 100).round(2)
        
        # Classify Severity Labels
        conditions = [
            (df['pei_score'] <= 40.0),
            (df['pei_score'] > 40.0) & (df['pei_score'] <= 70.0),
            (df['pei_score'] > 70.0)
        ]
        choices = ['Moderate', 'High', 'Critical']
        df['severity_label'] = np.select(conditions, choices, default='Moderate')
        
        # Sort by PEI to generate leaderboard
        self.leaderboard = df.sort_values(by="pei_score", ascending=False).reset_index(drop=True)
        
        # Save PEI leaderboard to reports
        self.leaderboard.to_json(REPORTS_DIR / "pei_leaderboard.json", orient="records", indent=4)
        
        return self.leaderboard

    def get_summary_stats(self):
        """Return counts and distributions of PEI categories."""
        if self.leaderboard is None or self.leaderboard.empty:
            return {}
            
        stats = {
            "total_hotspots": len(self.leaderboard),
            "critical_count": int((self.leaderboard['severity_label'] == 'Critical').sum()),
            "high_count": int((self.leaderboard['severity_label'] == 'High').sum()),
            "moderate_count": int((self.leaderboard['severity_label'] == 'Moderate').sum()),
            "average_pei": float(self.leaderboard['pei_score'].mean()),
            "max_pei": float(self.leaderboard['pei_score'].max()),
        }
        return stats

if __name__ == "__main__":
    from app.data_pipeline import DataPipeline
    from services.hotspot_service import HotspotService
    
    dp = DataPipeline()
    df = dp.load_data()
    hs = HotspotService()
    _, summary = hs.run_spatial_clustering(df)
    
    pei_serv = PEIService(summary)
    leaderboard = pei_serv.calculate_pei()
    print("PEI Leaderboard:")
    print(leaderboard[['cluster_id', 'pei_score', 'severity_label', 'violation_count']].head(5))
