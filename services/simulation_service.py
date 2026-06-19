import os
import json
import logging
from pathlib import Path
from app.config import SIMULATIONS_DIR

logger = logging.getLogger("ParkTwinAI.SimulationService")

class SimulationService:
    def __init__(self):
        # Sustainability factors
        self.IDLE_FUEL_CONSUMPTION_L_PER_HOUR = 1.2  # 1.2 liters of fuel wasted per hour of idling
        self.CO2_KG_PER_LITER = 2.31                # 2.31 kg CO2 produced per liter of gasoline
        self.VALUE_OF_TIME_PER_HOUR_USD = 15.00     # $15/hour average value of passenger time
        self.FUEL_COST_PER_LITER_USD = 1.15         # $1.15 per liter of fuel

    def get_intervention_params(self, intervention_type):
        """Return reduction ratios for frequency and duration for each intervention."""
        # Returns: (frequency_reduction_ratio, duration_reduction_ratio)
        params = {
            "Tow Vehicle": (0.0, 0.80),         # 80% duration reduction (towed quickly)
            "Fine Only": (0.30, 0.0),           # 30% frequency reduction
            "Monitor": (0.0, 0.10),             # 10% duration reduction
            "Increased Patrol": (0.50, 0.20),    # 50% frequency, 20% duration
            "Officer Patrol": (0.50, 0.25),     # 50% frequency, 25% duration
            "Barricading": (0.40, 0.10),        # 40% frequency, 10% duration
            "No Intervention": (0.0, 0.0)       # 0% reduction
        }
        return params.get(intervention_type, (0.0, 0.0))

    def run_simulation(self, base_delay_vhl, intervention_type, hotspot_name="Hotspot Area"):
        """Simulate intervention impact on vehicle-hours lost (VHL) and calculate savings."""
        freq_red, dur_red = self.get_intervention_params(intervention_type)
        
        # Calculate combined delay reduction
        # Delay = Frequency * Duration * Baseline Delay
        # Projected Delay = (1 - freq_red) * (1 - dur_red) * Base Delay
        reduction_multiplier = (1.0 - freq_red) * (1.0 - dur_red)
        projected_delay_vhl = round(base_delay_vhl * reduction_multiplier, 2)
        delay_prevented_vhl = round(base_delay_vhl - projected_delay_vhl, 2)
        
        improvement_percentage = round((delay_prevented_vhl / base_delay_vhl * 100.0), 2) if base_delay_vhl > 0 else 0.0
        
        # Calculate Sustainability Metrics
        fuel_saved_liters = round(delay_prevented_vhl * self.IDLE_FUEL_CONSUMPTION_L_PER_HOUR, 2)
        co2_avoided_kg = round(fuel_saved_liters * self.CO2_KG_PER_LITER, 2)
        
        # Economic impact
        time_cost_saved = delay_prevented_vhl * self.VALUE_OF_TIME_PER_HOUR_USD
        fuel_cost_saved = fuel_saved_liters * self.FUEL_COST_PER_LITER_USD
        total_dollars_saved = round(time_cost_saved + fuel_cost_saved, 2)
        
        # Generate 24-hour simulation profiles
        # Dual-peak hourly traffic distribution (peaks at 9 AM and 6 PM)
        hourly_weights = [
            0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 
            0.08, 0.12, 0.10, 0.06, 0.05, 0.04,  # Morning Peak (7-9 AM)
            0.04, 0.05, 0.04, 0.05, 0.08, 0.12,  # Evening Peak (5-6 PM)
            0.10, 0.06, 0.04, 0.02, 0.02, 0.01
        ]
        
        hourly_before = [round(base_delay_vhl * weight, 3) for weight in hourly_weights]
        hourly_after = [round(projected_delay_vhl * weight, 3) for weight in hourly_weights]
        
        simulation_report = {
            "hotspot_name": hotspot_name,
            "intervention": intervention_type,
            "metrics": {
                "base_delay_vhl": base_delay_vhl,
                "projected_delay_vhl": projected_delay_vhl,
                "delay_prevented_vhl": delay_prevented_vhl,
                "improvement_percentage": improvement_percentage,
                "fuel_saved_liters": fuel_saved_liters,
                "co2_avoided_kg": co2_avoided_kg,
                "dollars_saved": total_dollars_saved
            },
            "hourly_profile": {
                "before": hourly_before,
                "after": hourly_after
            },
            "assumptions": {
                "idling_fuel_consumption_rate_l_hr": self.IDLE_FUEL_CONSUMPTION_L_PER_HOUR,
                "co2_emissions_factor_kg_l": self.CO2_KG_PER_LITER,
                "value_of_time_usd_hr": self.VALUE_OF_TIME_PER_HOUR_USD,
                "fuel_cost_usd_l": self.FUEL_COST_PER_LITER_USD
            }
        }
        
        # Save report
        safe_name = hotspot_name.lower().replace(" ", "_")
        report_path = SIMULATIONS_DIR / f"sim_{safe_name}_{intervention_type.lower().replace(' ', '_')}.json"
        with open(report_path, 'w') as f:
            json.dump(simulation_report, f, indent=4)
            
        logger.info(f"Saved simulation results to {report_path}")
        return simulation_report

if __name__ == "__main__":
    sim = SimulationService()
    res = sim.run_simulation(150.0, "Increased Patrol", "Cubbon Park Road")
    print("Simulation Results:")
    print(res["metrics"])
