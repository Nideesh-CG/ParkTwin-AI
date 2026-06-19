import logging
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from app.config import SIMULATIONS_DIR

logger = logging.getLogger("ParkTwinAI.DelayService")

class DelayService:
    def __init__(self):
        # Empirical constants for congestion delay estimation
        self.BASE_DELAY_PER_MIN = 0.12  # 0.12 min delay per violation min on regular roads
        self.JUNCTION_MULTIPLIER = 2.5   # Junction parking is 2.5x more severe
        self.TRAFFIC_FLOW_DEFAULT = 600 # 600 vehicles/hour baseline traffic flow

    def estimate_causal_delay(self, pei_score, duration_minutes, is_junction):
        """Estimate traffic minutes lost and vehicle-hours lost (VHL) caused by a violation.
        
        Assumptions:
        1. Baseline delay propagates based on duration of blockage.
        2. Proximity to a junction amplifies delay due to merging conflicts and turning blocks.
        3. High PEI scores represent higher traffic volume and compounding density.
        4. Vehicle-Hours Lost (VHL) = (Delay per vehicle in hours) * (Traffic volume passing by during violation).
        """
        # Junction factor
        junc_factor = self.JUNCTION_MULTIPLIER if is_junction else 1.0
        
        # PEI compounding factor
        pei_factor = 1.0 + (pei_score / 50.0)
        
        # Compute minutes of delay per affected vehicle
        delay_per_vehicle_min = duration_minutes * self.BASE_DELAY_PER_MIN * junc_factor * pei_factor
        
        # Total vehicles affected during the duration
        duration_hours = duration_minutes / 60.0
        total_vehicles_affected = self.TRAFFIC_FLOW_DEFAULT * duration_hours
        
        # Calculate Vehicle-Hours Lost (VHL)
        # VHL = (Delay per vehicle in mins / 60) * total vehicles affected
        vehicle_hours_lost = (delay_per_vehicle_min / 60.0) * total_vehicles_affected
        
        # Cap unreasonable numbers
        delay_per_vehicle_min = round(min(delay_per_vehicle_min, 180.0), 2)
        vehicle_hours_lost = round(min(vehicle_hours_lost, 500.0), 2)
        
        return {
            "minutes_lost_per_vehicle": delay_per_vehicle_min,
            "vehicles_affected": int(total_vehicles_affected),
            "vehicle_hours_lost": vehicle_hours_lost,
            "assumptions": [
                f"Baseline traffic flow set to {self.TRAFFIC_FLOW_DEFAULT} vehicles/hour",
                f"Junction criticality multiplier is {junc_factor}x",
                f"PEI risk factor scaled delay by {pei_factor:.2f}x"
            ]
        }

    def analyze_ripple_effect(self, primary_delay_vhl, hotspot_name="Primary Junction"):
        """Construct local intersection network and estimate delay propagation using NetworkX."""
        # Create a lightweight synthetic grid network around the hotspot
        G = nx.DiGraph()
        
        # Nodes representing junctions
        # 0: Primary hotspot
        # 1, 2, 3: 1st hop neighbors (Secondary Zone)
        # 4, 5, 6: 2nd hop neighbors (Tertiary Zone)
        G.add_node(0, label=hotspot_name, zone="Primary", delay=primary_delay_vhl)
        
        # Add 1st hop neighbors (Secondary zone, 50% delay propagation)
        sec_delay = round(primary_delay_vhl * 0.50, 2)
        G.add_node(1, label="North Intersection", zone="Secondary", delay=sec_delay)
        G.add_node(2, label="East Crossroad", zone="Secondary", delay=sec_delay)
        G.add_node(3, label="West Bypass", zone="Secondary", delay=sec_delay)
        
        # Add 2nd hop neighbors (Tertiary/Low zone, 20% delay propagation)
        tert_delay = round(primary_delay_vhl * 0.20, 2)
        G.add_node(4, label="Highway Entry", zone="Low", delay=tert_delay)
        G.add_node(5, label="Market Circle", zone="Low", delay=tert_delay)
        G.add_node(6, label="Residential Exit", zone="Low", delay=tert_delay)
        
        # Add directed edges (representing street flows)
        G.add_edge(0, 1, weight=300) # 300 meters
        G.add_edge(0, 2, weight=400)
        G.add_edge(0, 3, weight=350)
        G.add_edge(1, 4, weight=500)
        G.add_edge(2, 5, weight=600)
        G.add_edge(3, 6, weight=450)
        
        # Compiling Affected Junctions List
        affected_junctions = []
        total_spillover_delay = 0.0
        
        for node in G.nodes():
            data = G.nodes[node]
            affected_junctions.append({
                "node_id": int(node),
                "name": data["label"],
                "zone": data["zone"],
                "allocated_delay_vhl": float(data["delay"])
            })
            if data["zone"] != "Primary":
                total_spillover_delay += data["delay"]
                
        # Generate Network Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = {
            0: (0, 0),
            1: (0, 2),
            2: (2, -1),
            3: (-2, -1),
            4: (1, 4),
            5: (4, -2),
            6: (-4, -2)
        }
        
        # Colors based on zones
        color_map = []
        for node in G:
            zone = G.nodes[node]["zone"]
            if zone == "Primary":
                color_map.append("#E53935") # Dark Red
            elif zone == "Secondary":
                color_map.append("#FB8C00") # Orange
            else:
                color_map.append("#4CAF50") # Green
                
        labels = {node: f"{G.nodes[node]['label']}\n{G.nodes[node]['delay']} VHL" for node in G}
        
        nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=2000, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, width=2, arrowstyle='->', arrowsize=20, edge_color='#B0BEC5', ax=ax)
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="#000000", font_weight="bold", ax=ax)
        
        plt.title(f"Congestion Ripple Effect Propagation Network from {hotspot_name}", fontsize=12, pad=15)
        plt.axis("off")
        plt.tight_layout()
        
        network_plot_path = SIMULATIONS_DIR / "network_ripple_effect.png"
        plt.savefig(network_plot_path, dpi=150)
        plt.close()
        
        return {
            "network": G,
            "primary_impact_zone": hotspot_name,
            "secondary_impact_zone": ["North Intersection", "East Crossroad", "West Bypass"],
            "affected_junctions": affected_junctions,
            "estimated_spillover_delay_vhl": round(total_spillover_delay, 2),
            "network_plot_path": str(network_plot_path)
        }

    def generate_ripple_frames(self, primary_delay_vhl, hotspot_name="Primary Junction"):
        """Generate three sequential frames representing congestion ripple propagation."""
        frames = []
        for stage in [1, 2, 3]:
            G = nx.DiGraph()
            
            # Node 0 (Primary)
            G.add_node(0, label=hotspot_name, zone="Primary", delay=primary_delay_vhl if stage >= 1 else 0.0)
            
            # Nodes 1, 2, 3 (Secondary)
            sec_delay = round(primary_delay_vhl * 0.50, 2) if stage >= 2 else 0.0
            G.add_node(1, label="North Intersection", zone="Secondary", delay=sec_delay)
            G.add_node(2, label="East Crossroad", zone="Secondary", delay=sec_delay)
            G.add_node(3, label="West Bypass", zone="Secondary", delay=sec_delay)
            
            # Nodes 4, 5, 6 (Tertiary)
            tert_delay = round(primary_delay_vhl * 0.20, 2) if stage >= 3 else 0.0
            G.add_node(4, label="Highway Entry", zone="Low", delay=tert_delay)
            G.add_node(5, label="Market Circle", zone="Low", delay=tert_delay)
            G.add_node(6, label="Residential Exit", zone="Low", delay=tert_delay)
            
            G.add_edge(0, 1, weight=300)
            G.add_edge(0, 2, weight=400)
            G.add_edge(0, 3, weight=350)
            G.add_edge(1, 4, weight=500)
            G.add_edge(2, 5, weight=600)
            G.add_edge(3, 6, weight=450)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            pos = {0: (0, 0), 1: (0, 2), 2: (2, -1), 3: (-2, -1), 4: (1, 4), 5: (4, -2), 6: (-4, -2)}
            
            color_map = []
            for node in G:
                zone = G.nodes[node]["zone"]
                delay_val = G.nodes[node]["delay"]
                if zone == "Primary" and delay_val > 0:
                    color_map.append("#E53935") # Red
                elif zone == "Secondary" and delay_val > 0:
                    color_map.append("#FB8C00") # Orange
                elif zone == "Low" and delay_val > 0:
                    color_map.append("#10B981") # Green
                else:
                    color_map.append("#9CA3AF") # Muted Gray (inactive)
                    
            labels = {node: f"{G.nodes[node]['label']}\n{G.nodes[node]['delay']} VHL" for node in G}
            
            nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=2000, alpha=0.9, ax=ax)
            nx.draw_networkx_edges(G, pos, width=2, arrowstyle='->', arrowsize=20, edge_color='#B0BEC5', ax=ax)
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="#000000", font_weight="bold", ax=ax)
            
            plt.title(f"Congestion Ripple Effect - Propagation Stage {stage}/3", fontsize=11, pad=10)
            plt.axis("off")
            plt.tight_layout()
            
            frame_path = SIMULATIONS_DIR / f"network_ripple_stage_{stage}.png"
            plt.savefig(frame_path, dpi=150)
            plt.close()
            frames.append(str(frame_path))
            
        return frames


if __name__ == "__main__":
    ds = DelayService()
    delay = ds.estimate_causal_delay(pei_score=85, duration_minutes=120, is_junction=True)
    print("Estimated Delay Details:")
    print(delay)
    ripple = ds.analyze_ripple_effect(delay["vehicle_hours_lost"], "Modi Hospital Junction")
    print("\nAffected Junctions:")
    print(ripple["affected_junctions"])
    print("Spillover delay:", ripple["estimated_spillover_delay_vhl"])
