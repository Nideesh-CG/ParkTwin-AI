import logging
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from app.config import DBSCAN_EPS_KM, DBSCAN_MIN_SAMPLES, EARTH_RADIUS_KM, MAX_SAMPLE_SIZE, HEATMAPS_DIR

logger = logging.getLogger("ParkTwinAI.HotspotService")

class HotspotService:
    def __init__(self, df=None):
        self.df = df
        self.clustered_df = None
        self.top_hotspots_summary = None

    def sample_data_if_needed(self, df, max_rows=MAX_SAMPLE_SIZE):
        """Sample data to keep calculations fast and avoid memory overflow."""
        if len(df) > max_rows:
            logger.info(f"Dataset has {len(df)} rows. Sampling down to {max_rows} rows for clustering.")
            return df.sample(n=max_rows, random_state=42).copy()
        return df.copy()

    def run_spatial_clustering(self, df=None):
        """Run DBSCAN spatial clustering on latitude and longitude coordinates."""
        if df is not None:
            self.df = df
            
        if self.df is None or len(self.df) == 0:
            raise ValueError("No dataframe loaded for clustering.")
            
        working_df = self.sample_data_if_needed(self.df)
        
        # DBSCAN on spatial coordinates
        coords = np.radians(working_df[['latitude', 'longitude']].values)
        
        # Convert eps (km) to radians
        eps_rad = DBSCAN_EPS_KM / EARTH_RADIUS_KM
        
        db = DBSCAN(eps=eps_rad, min_samples=DBSCAN_MIN_SAMPLES, metric='haversine', algorithm='ball_tree')
        labels = db.fit_predict(coords)
        
        working_df['cluster_id'] = labels
        self.clustered_df = working_df
        
        # Calculate cluster summaries (excluding noise cluster -1)
        clusters = working_df[working_df['cluster_id'] != -1]
        
        if len(clusters) == 0:
            logger.warning("DBSCAN did not find any spatial clusters. Setting default hotspots.")
            # Create dummy cluster 0 with all data
            working_df['cluster_id'] = 0
            clusters = working_df
            
        cluster_groups = clusters.groupby('cluster_id')
        
        summary_list = []
        for cid, group in cluster_groups:
            # Spatial center
            center_lat = group['latitude'].mean()
            center_lng = group['longitude'].longitude = group['longitude'].mean()
            count = len(group)
            
            # Key statistics
            avg_duration = group['duration_minutes'].mean()
            top_vehicle = group['vehicle_type'].mode().iloc[0] if not group['vehicle_type'].mode().empty else "UNKNOWN"
            top_station = group['police_station'].mode().iloc[0] if not group['police_station'].mode().empty else "Unknown"
            top_junction = group['junction'].mode().iloc[0] if not group['junction'].mode().empty else "No Junction"
            
            # Temporal breakdown
            hours = group['created_datetime'].dt.hour
            peak_count = ((hours >= 8) & (hours <= 11) | (hours >= 17) & (hours <= 20)).sum()
            peak_ratio = peak_count / count if count > 0 else 0
            
            summary_list.append({
                "cluster_id": int(cid),
                "latitude": float(center_lat),
                "longitude": float(center_lng),
                "violation_count": int(count),
                "avg_duration_minutes": float(avg_duration),
                "primary_vehicle_type": top_vehicle,
                "primary_police_station": top_station,
                "primary_junction": top_junction,
                "peak_ratio": float(peak_ratio)
            })
            
        summary_df = pd.DataFrame(summary_list)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(by="violation_count", ascending=False).reset_index(drop=True)
            # Add ranking
            summary_df['rank'] = summary_df.index + 1
        
        self.top_hotspots_summary = summary_df
        
        # Save summary to heatmaps directory
        summary_df.to_json(HEATMAPS_DIR / "hotspots_summary.json", orient="records", indent=4)
        
        return working_df, summary_df

    def get_temporal_breakdowns(self, df=None):
        """Categorize violations by time-of-day and weekday/weekend."""
        target_df = df if df is not None else self.df
        if target_df is None or len(target_df) == 0:
            raise ValueError("No dataframe available.")
            
        df_copy = target_df.copy()
        
        # Time-of-Day Category
        hours = df_copy['created_datetime'].dt.hour
        conditions = [
            (hours >= 6) & (hours < 12),
            (hours >= 12) & (hours < 17),
            (hours >= 17) & (hours < 21),
            (hours < 6) | (hours >= 21)
        ]
        choices = ['Morning', 'Afternoon', 'Evening', 'Night']
        df_copy['time_of_day'] = np.select(conditions, choices, default='Night')
        
        # Weekly Category
        days = df_copy['created_datetime'].dt.dayofweek
        df_copy['day_type'] = np.where(days < 5, 'Weekday', 'Weekend')
        
        tod_counts = df_copy['time_of_day'].value_counts().to_dict()
        weekly_counts = df_copy['day_type'].value_counts().to_dict()
        
        return df_copy, tod_counts, weekly_counts

if __name__ == "__main__":
    from app.data_pipeline import DataPipeline
    dp = DataPipeline()
    df = dp.load_data()
    hs = HotspotService()
    clustered, summary = hs.run_spatial_clustering(df)
    print("Top 5 Hotspots:")
    print(summary.head(5))
