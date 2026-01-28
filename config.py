"""
Configuration module for microkinetic modeling parameters and settings.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class SolverSettings:
    """Configuration class for solver parameters."""

    # pH and potential ranges
    pH_list: List[float] = field(default_factory=lambda: [13])
    V_list: List[float] = field(default_factory=lambda: [0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1])
    
    # Simulation parameters
    temperature: float = 298  # K
    time: float = 1e5  # seconds
    abstol: float = 1e-20
    reltol: float = 1e-10

    # NEW: Sweep functionality
    enable_sweep_mode: bool = False
    sweep_rate: float = 0.1  # V/s (e.g., 100 mV/s = 0.1 V/s)
    use_coverage_propagation: bool = True  # Use coverage from previous step

    # File paths
    input_excel_path: str = "input.xlsx"
    executable_path: str = "D:\mkmcxx\mkmcxx-2.15.3-windows-x64\mkmcxx_2.15.3\bin\mkmcxx.exe"  # To be set by user

    # Reaction parameters
    pre_exponential_factor: float = 6.21e12

    # Plotting parameters
    site_density: float = 2.94e-5  # mol sites/m^2
    target_species: List[str] = field(default_factory=lambda: ['CH3OH', 'CH4', 'CO', 'HCOOH', 'H2', 'CH2O'])
    species_electrons: Dict[str, int] = field(default_factory=lambda: {
        'CH3OH': 6, 'CH4': 8, 'CO': 2, 'HCOOH': 2, 'H2': 2, 'CH2O': 4
    })    

    # Output settings
    output_base_dir: str = "results"
    create_plots: bool = True
    plot_format: str = "png"

    # def calculate_step_time(self) -> float:
    #     """
    #     Compute time per step based on sweep_rate and uniform ΔV.
    #     Returns default `time` if sweep disabled or V_list invalid.
    #     """
    #     if not self.enable_sweep_mode or len(self.V_list) < 2:
    #         return self.time

    #     # Force all entries to floats
    #     try:
    #         Vs: List[float] = [float(v) for v in sum(
    #             ([x] if not isinstance(x, list) else x for x in self.V_list), []
    #         )]
    #     except Exception as e:
    #         # Fallback on naive cast of top‐level items
    #         Vs = [float(v) for v in self.V_list]

    #     # Sort by absolute magnitude
    #     Vs_sorted = sorted(Vs, key=abs)

    #     # Compute uniform step
    #     dv = abs(Vs_sorted[1] - Vs_sorted)
    #     if self.sweep_rate == 0:
    #         raise ValueError("sweep_rate must be non‐zero")
    #     return dv / self.sweep_rate
    
    def calculate_step_time(self) -> float:
        """
        Calculate time per potential step for sweep mode.
        Returns default time if sweep mode is disabled.
        """
        if not self.enable_sweep_mode or len(self.V_list) < 2:
            return self.time

        try:
            # Ensure V_list is properly flattened and converted to floats
            Vs = [float(v) for v in self.V_list]

            # Sort by absolute value for proper sweep order
            Vs_sorted = sorted(Vs, key=abs)

            # Calculate uniform step size
            dv = abs(Vs_sorted[1] - Vs_sorted[0])  # Fixed: subtract two floats, not float - list

            if dv == 0:
                logger.warning("Potential step size is zero, using default time")
                return self.time

            if self.sweep_rate == 0:
                raise ValueError("sweep_rate must be non-zero for sweep mode")

            step_time = dv / self.sweep_rate
            logger.debug(f"Calculated step time: {step_time} s (dv={dv} V, rate={self.sweep_rate} V/s)")
            return step_time

        except Exception as e:
            logger.error(f"Error calculating step time: {e}")
            logger.debug(f"V_list content: {self.V_list}, types: {[type(v) for v in self.V_list]}")
            return self.time

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SolverSettings':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> 'SolverSettings':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.__dict__.copy()
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def to_json(self, json_path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = self.__dict__.copy()
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []

        if not self.pH_list:
            errors.append("pH_list cannot be empty")

        if not self.V_list:
            errors.append("V_list cannot be empty")

        if self.temperature <= 0:
            errors.append("Temperature must be positive")

        if self.time <= 0:
            errors.append("Time must be positive")

        if not Path(self.input_excel_path).exists():
            errors.append(f"Input Excel file not found: {self.input_excel_path}")

        return errors

# Default configuration instance
DEFAULT_CONFIG = SolverSettings()

def load_config(config_path: Optional[str] = None) -> SolverSettings:
    """Load configuration from file or return default."""
    if config_path is None:
        return DEFAULT_CONFIG

    path = Path(config_path)
    if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
        return SolverSettings.from_yaml(config_path)
    elif path.suffix.lower() == '.json':
        return SolverSettings.from_json(config_path)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")
    

