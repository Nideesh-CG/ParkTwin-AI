import logging
import pandas as pd
import numpy as np
from services.delay_service import DelayService

logger = logging.getLogger("ParkTwinAI.ImmunityService")

class ImmunityService:
    def __init__(self, leaderboard_df=None, memory_profiles=None):
        self.leaderboard = leaderboard_df
        self.memory_profiles = memory_profiles
        self.delay_service = DelayService()
        self.immunity_data = {}

    def calculate_immunity_scores(self):
        """Calculate Traffic Immunity Score (TIS) for all hotspots."""
        if self.leaderboard is None or self.leaderboard.empty or not self.memory_profiles:
            logger.warning("Empty dependencies. Cannot calculate immunity scores.")
            return {}
            
        immunity_profiles = {}
        
        # Max values for normalization
        max_duration = max(float(row.get('avg_duration_minutes', 60)) for _, row in self.leaderboard.iterrows()) or 60.0
        max_violations = max(int(row.get('violation_count', 10)) for _, row in self.leaderboard.iterrows()) or 10.0
        
        for _, row in self.leaderboard.iterrows():
            junction = row['primary_junction']
            if junction not in self.memory_profiles:
                continue
                
            mem = self.memory_profiles[junction]
            
            # 1. Recovery Speed (0-100)
            # Shorter Tow recovery time yields higher recovery speed. Max recovery time is max_duration.
            tow_rec_time = mem["avg_recovery_time"]
            recovery_speed = max(0.0, 100.0 - (tow_rec_time / max_duration * 100.0))
            
            # 2. Intervention Effectiveness (0-100)
            # success rate of Tow vehicle (approx 78%)
            intervention_effectiveness = mem["best_effectiveness"]
            
            # 3. Spillover Resistance (0-100)
            # Estimate baseline delay VHL, run NetworkX ripple engine, and compute resistance to spillover
            is_junction = junction != "No Junction"
            # Estimate primary delay using DelayService
            est_delay_dict = self.delay_service.estimate_causal_delay(
                pei_score=float(row['pei_score']),
                duration_minutes=float(row['avg_duration_minutes']),
                is_junction=is_junction
            )
            primary_delay_vhl = est_delay_dict["vehicle_hours_lost"]
            
            ripple_results = self.delay_service.analyze_ripple_effect(primary_delay_vhl, hotspot_name=junction)
            spillover_vhl = ripple_results["estimated_spillover_delay_vhl"]
            
            # Lower spillover delay means higher resistance. Normalize relative to primary delay.
            # Max possible spillover is 2.1 * primary_delay
            max_spillover = max(primary_delay_vhl * 2.1, 1.0)
            spillover_resistance = max(0.0, 100.0 - (spillover_vhl / max_spillover * 100.0))
            
            # 4. Recurrence Resistance (0-100)
            # Fewer violations = higher resistance.
            violation_count = mem["recurrence_count"]
            recurrence_resistance = max(0.0, 100.0 - (violation_count / max_violations * 100.0))
            
            # 5. Sustainability Efficiency (0-100)
            # Simulating potential CO2 avoided via Tow vehicle deployment.
            # 80% duration reduction
            delay_prevented = primary_delay_vhl * 0.80
            fuel_saved_l = delay_prevented * 1.2
            co2_avoided_kg = fuel_saved_l * 2.31
            
            # Scale relative to 1500 kg CO2 limit
            sustainability_efficiency = min(100.0, (co2_avoided_kg / 1500.0) * 100.0)
            
            # TIS Weighted Formula:
            # 0.30 * Recovery Speed + 0.25 * Intervention Effectiveness + 0.20 * Spillover Resistance + 0.15 * Recurrence Resistance + 0.10 * Sustainability Efficiency
            tis = (
                0.30 * recovery_speed +
                0.25 * intervention_effectiveness +
                0.20 * spillover_resistance +
                0.15 * recurrence_resistance +
                0.10 * sustainability_efficiency
            )
            
            tis = round(float(np.clip(tis, 0, 100)), 1)
            
            # Classify
            if tis <= 40.0:
                classification = "Fragile"
                color_badge = "red"
            elif tis <= 70.0:
                classification = "Adaptive"
                color_badge = "orange"
            else:
                classification = "Resilient"
                color_badge = "green"
                
            immunity_profiles[junction] = {
                "junction": junction,
                "tis_score": tis,
                "classification": classification,
                "color_badge": color_badge,
                "metrics": {
                    "recovery_speed": round(recovery_speed, 1),
                    "intervention_effectiveness": round(intervention_effectiveness, 1),
                    "spillover_resistance": round(spillover_resistance, 1),
                    "recurrence_resistance": round(recurrence_resistance, 1),
                    "sustainability_efficiency": round(sustainability_efficiency, 1)
                }
            }
            
        self.immunity_data = immunity_profiles
        logger.info(f"Calculated Traffic Immunity Scores for {len(immunity_profiles)} hotspots.")
        return immunity_profiles
        
    def estimate_simulated_immunity(self, junction_name, intervention_type):
        """Estimate the projected immunity score improvement after deploying an intervention."""
        if not self.immunity_data or junction_name not in self.immunity_data:
            return None
            
        base_imm = self.immunity_data[junction_name]
        tis_before = base_imm["tis_score"]
        
        # Simulating changes:
        # Tow vehicle yields highest improvement: duration reduced by 80%, recovery speed goes up, spillover reduced.
        # Fine Only: 30% frequency reduction, 0% duration reduction.
        # Patrol: 50% frequency, 25% duration.
        
        params = {
            "Tow Vehicle": (0.80, 0.80, 0.50, 0.80), # (recovery_speed_factor, effectiveness, recurrence_reduction, sustainability_factor)
            "Officer Patrol": (0.45, 0.45, 0.30, 0.50),
            "Barricades": (0.35, 0.35, 0.20, 0.40),
            "Fine Only": (0.15, 0.20, 0.15, 0.20),
            "No Intervention": (0.0, 0.0, 0.0, 0.0)
        }
        
        speed_factor, eff_multiplier, rec_reduction, sust_factor = params.get(intervention_type, (0.0, 0.0, 0.0, 0.0))
        
        # Calculate new components
        # 1. Recovery speed improves (closer to 100)
        rs_before = base_imm["metrics"]["recovery_speed"]
        rs_after = rs_before + (100.0 - rs_before) * speed_factor
        
        # 2. Effectiveness becomes the chosen intervention's rate
        # Best rate is best_effectiveness, others are lower
        ie_after = base_imm["metrics"]["intervention_effectiveness"] * (0.5 + 0.5 * eff_multiplier)
        
        # 3. Spillover resistance improves as delay decreases
        sr_before = base_imm["metrics"]["spillover_resistance"]
        sr_after = sr_before + (100.0 - sr_before) * speed_factor
        
        # 4. Recurrence resistance improves
        rr_before = base_imm["metrics"]["recurrence_resistance"]
        rr_after = rr_before + (100.0 - rr_before) * rec_reduction
        
        # 5. Sustainability efficiency improves
        se_before = base_imm["metrics"]["sustainability_efficiency"]
        se_after = se_before + (100.0 - se_before) * sust_factor
        
        tis_after = (
            0.30 * rs_after +
            0.25 * ie_after +
            0.20 * sr_after +
            0.15 * rr_after +
            0.10 * se_after
        )
        tis_after = round(float(np.clip(tis_after, 0, 100)), 1)
        
        # Re-classify
        if tis_after <= 40.0:
            class_after = "Fragile"
        elif tis_after <= 70.0:
            class_after = "Adaptive"
        else:
            class_after = "Resilient"
            
        improvement = round(tis_after - tis_before, 1)
        
        return {
            "tis_before": tis_before,
            "class_before": base_imm["classification"],
            "tis_after": tis_after,
            "class_after": class_after,
            "improvement": improvement
        }
