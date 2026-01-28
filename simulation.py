"""
Optimized simulation runner that uses cached Excel data.
Complete standalone version.
"""

import os
import subprocess
import shutil
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Iterable, Tuple
from itertools import zip_longest
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimulationParameters:
    """Container for simulation parameters."""
    temperature: float
    potential: float
    time: float
    abstol: float
    reltol: float
    pressure: float
    pH: float
    pre_exponential_factor: float = 6.21e12


class CoverageManager:
    """Manages coverage data between simulation steps."""

    def __init__(self):
        self.coverage_data: Dict[float, Dict[float, Dict[str, float]]] = {}

    def save_coverage(self, pH: float, V: float, coverage_dict: Dict[str, float]) -> None:
        """Save coverage data for a specific pH/V combination."""
        if pH not in self.coverage_data:
            self.coverage_data[pH] = {}
        self.coverage_data[pH][V] = dict(coverage_dict)
        logger.debug(f"Saved coverage for pH={pH}, V={V}")

    def get_coverage(self, pH: float, V: float) -> Optional[Dict[str, float]]:
        """Get coverage data for a specific pH/V combination."""
        return self.coverage_data.get(pH, {}).get(V, None)

    def get_previous_coverage(self, pH: float, V_current: float, V_list: List[float]) -> Optional[Dict[str, float]]:
        """Get coverage from the previous potential step."""
        V_sorted = sorted(V_list, key=lambda v: abs(v))
        try:
            current_idx = V_sorted.index(V_current)
            if current_idx > 0:
                prev_V = V_sorted[current_idx - 1]
                return self.get_coverage(pH, prev_V)
        except ValueError:
            logger.warning(f"Current potential {V_current} not found in V_list")
        return None

    def export_coverage_trajectory(self, output_file: str) -> None:
        """Export coverage trajectory data to JSON."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.coverage_data, f, indent=2)
            logger.info(f"Coverage trajectory exported to {output_file}")
        except Exception as e:
            logger.error(f"Failed to export coverage trajectory: {e}")


class InputFileGenerator:
    """Generates input files for microkinetic simulations."""

    def __init__(self, executable_path: str = ""):
        self.executable_path = executable_path

    def generate_input_file(
        self,
        data: Dict[str, Any],
        sim_params: SimulationParameters,
        output_filename: str = "input_file.mkm"
    ) -> str:
        """Generate input file for microkinetic modeling."""
        try:
            logger.debug(f"Generating input file with pH={sim_params.pH}, V={sim_params.potential}")
            logger.debug(f"Data dict pH={data.get('pH')}, V={data.get('V')}")
            
            with open(output_filename, 'w') as f:
                self._write_compounds_section(f, data, sim_params)
                self._write_reactions_section(f, data, sim_params)
                self._write_settings_section(f, sim_params)
                self._write_runs_section(f, sim_params)
            logger.debug(f"Generated input file: {output_filename}")
            return output_filename
        except Exception as e:
            logger.error(f"Error generating input file: {e}")
            raise

    def _write_compounds_section(self, file, data: Dict[str, Any], sim_params: SimulationParameters) -> None:
        file.write('&compounds\n\n')

        # Gas-phase compounds
        file.write("#gas-phase compounds\n\n#Name; isSite; concentration\n\n")
        for compound, concentration in zip_longest(data['gases'], data['concentrations'], fillvalue=0.0):
            compound_name = compound.strip("{}")
            
            # Special handling for OH based on pH
            if compound_name == "OH":
                # Calculate OH concentration from pH
                pOH = 14 - sim_params.pH
                concentration = 10 ** (-pOH)
                logger.debug(f"  OH concentration calculated from pH={sim_params.pH}: {concentration:.2e}")
            elif compound_name == "H": 
                # Calculate H+ concentration from pH
                concentration = 10 ** (-sim_params.pH)
                logger.debug(f"  H concentration calculated from pH={sim_params.pH}: {concentration:.2e}")   
            
            file.write(f"{compound:<15}; 0; {concentration}\n")

        # Adsorbates
        file.write("\n\n#adsorbates\n\n#Name; isSite; activity\n\n")
        for compound, activity in zip(data['adsorbates'], data['activity']):
            file.write(f"{compound:<15}; 1; {activity}\n")

        # Free sites
        free_site_cov = data.get('free_site_coverage', 1.0)
        file.write("\n#free sites on the surface\n\n")
        file.write("#Name; isSite; activity\n\n")
        file.write(f"*; 1; {free_site_cov}\n\n")

    def _write_reactions_section(self, file, data: Dict[str, Any], sim_params: SimulationParameters) -> None:
        file.write('&reactions\n\n')
        reactions = data['reactions']
        pre_exp = float(sim_params.pre_exponential_factor)

        logger.debug(f"Writing {len(reactions)} reactions")
        if len(reactions) > 0:
            logger.debug(f"First Ea type: {type(data['Ea'][0])}, value: {data['Ea'][0]}")
            logger.debug(f"First Eb type: {type(data['Eb'][0])}, value: {data['Eb'][0]}")
            logger.debug(f"Pre-exp type: {type(pre_exp)}, value: {pre_exp}")

        for j in range(len(reactions)):
            r1, r2, r3 = data['Reactant1'][j], data['Reactant2'][j], data['Reactant3'][j]
            p1, p2, p3 = data['Product1'][j], data['Product2'][j], data['Product3'][j]
            
            # Ensure Ea and Eb are floats
            try:
                ea = float(data['Ea'][j])
                eb = float(data['Eb'][j])
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting barriers for reaction {j}: {e}")
                logger.error(f"  Ea[{j}] = {data['Ea'][j]} (type: {type(data['Ea'][j])})")
                logger.error(f"  Eb[{j}] = {data['Eb'][j]} (type: {type(data['Eb'][j])})")
                raise
            
            line = self._format_reaction_line(r1, r2, r3, p1, p2, p3, pre_exp, ea, eb)
            file.write(line)

    def _format_reaction_line(self, r1: str, r2: str, r3: str,
                              p1: str, p2: str, p3: str,
                              pre_exp: float, ea: float, eb: float) -> str:
        # Ensure all numeric values are floats (full precision, no rounding)
        pre_exp = float(pre_exp)
        ea = float(ea)
        eb = float(eb)
        
        # Use full precision for ea and eb (no format specifier = Python's default float representation)
        if r3:
            if p3:
                return (f"AR; {r1:<15} + {r2:<15} + {r3:<5} => "
                        f"{p1:<15} + {p2:<15} + {p3:<7};{pre_exp:.2e} ; {pre_exp:.2e} ; "
                        f"{ea} ; {eb} \n")
            else:
                return (f"AR; {r1:<15} + {r2:<15} + {r3:<5} => "
                        f"{p1:<15} + {p2:<20};{pre_exp:.2e} ; {pre_exp:.2e} ; "
                        f"{ea} ; {eb} \n")
        elif r2:
            if p3:
                return (f"AR; {r1:<15} + {r2:<14} => {p1:<10} + {p2:<15} + {p3:<7};"
                        f"{pre_exp:.2e} ; {pre_exp:.2e} ; {ea} ; {eb} \n")
            elif p2:
                return (f"AR; {r1:<15} + {r2:<15} => {p1:<15} + {p2:<20};"
                        f"{pre_exp:.2e} ; {pre_exp:.2e} ; {ea} ; {eb} \n")
            else:
                return (f"AR; {r1:<15} + {r2:<15} => {p1:<15}{'':<23};"
                        f"{pre_exp:.2e} ; {pre_exp:.2e} ; {ea} ; {eb} \n")
        else:
            if p2:
                return (f"AR; {r1:<15} {'':<17} => {p1:<15} + {p2:<20};"
                        f"{pre_exp:.2e} ; {pre_exp:.2e} ; {ea} ; {eb} \n")
            else:
                return (f"AR; {r1:<15} {'':<17} => {p1:<15}{'':<23};"
                        f"{pre_exp:.2e} ; {pre_exp:.2e} ; {ea} ; {eb} \n")

    def _write_settings_section(self, file, sim_params: SimulationParameters) -> None:
        file.write("\n\n&settings\n")
        file.write("TYPE = SEQUENCERUN\n")
        file.write("USETIMESTAMP = 0\n")
        file.write(f"PRESSURE = {sim_params.pressure}\n")
        file.write("POTAXIS=1\n")
        file.write("DEBUG=0\n")
        file.write("NETWORK_RATES=1\n")
        file.write("NETWORK_FLUX=1\n")

    def _write_runs_section(self, file, sim_params: SimulationParameters) -> None:
        file.write('\n\n&runs\n')
        file.write("# Temp; Potential;Time;AbsTol;RelTol\n")
        line = f"{sim_params.temperature:<5};{sim_params.potential:<5};{sim_params.time:<5.2e};{sim_params.abstol:<5};{sim_params.reltol:<5}"
        file.write(line)
        logger.debug(f"  Written to &runs: T={sim_params.temperature}, V={sim_params.potential}, time={sim_params.time}")

    def run_simulation(self, input_filename: str) -> subprocess.CompletedProcess:
        if not self.executable_path:
            raise ValueError("Executable path must be specified")

        if not Path(self.executable_path).exists():
            raise FileNotFoundError(f"Executable not found: {self.executable_path}")

        command = [self.executable_path, '-i', input_filename]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logger.debug("Simulation completed successfully")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Simulation failed: {e}")
            raise


class OptimizedSimulationRunner:
    """
    Optimized simulation runner that reads Excel once.
    Uses cached data processor for all subsequent operations.
    """

    EPS: float = 1e-9
    ENFORCE_SITE_BALANCE: bool = True
    MAX_COVERAGE: float = 1.0

    def __init__(self, config, excel_processor):
        """
        Initialize with config and pre-loaded Excel processor.
        
        Args:
            config: Configuration object
            excel_processor: CachedExcelDataProcessor instance (already loaded)
        """
        self.config = config
        self.excel_processor = excel_processor
        self.generator = InputFileGenerator(config.executable_path)
        self.coverage_manager = CoverageManager()

        # Allow config overrides
        self.EPS = getattr(config, "coverage_epsilon", self.EPS)
        self.ENFORCE_SITE_BALANCE = getattr(config, "enforce_site_balance", self.ENFORCE_SITE_BALANCE)
        self.MAX_COVERAGE = getattr(config, "max_coverage", self.MAX_COVERAGE)

    def _sanitize_value(self, x: float) -> float:
        """Clamp negative/near-zero values to zero, cap at MAX_COVERAGE."""
        try:
            if x < self.EPS:
                return 0.0
            if x > self.MAX_COVERAGE:
                return self.MAX_COVERAGE
            return x
        except Exception:
            return 0.0

    def _sanitize_mapping(self, cov: Dict[str, float]) -> Dict[str, float]:
        """Apply sanitization to all coverage values."""
        return {k: self._sanitize_value(float(v)) for k, v in cov.items()}

    def _renormalize_free_site(self, adsorbates: Iterable[str], activities: Iterable[float]) -> Tuple[List[float], float]:
        """Recompute free-site coverage: theta_* = max(0, 1 - sum(theta_i))."""
        act_list = [self._sanitize_value(a) for a in activities]
        total_ads = sum(act_list)
        theta_free = max(0.0, 1.0 - total_ads)
        theta_free = self._sanitize_value(theta_free)
        return act_list, theta_free

    def run_parameter_sweep(self, status_callback=None) -> None:
        """Run parameter sweep using cached Excel data."""
        base_dir = Path.cwd()
        results_dir = Path(self.config.output_base_dir)
        results_dir.mkdir(exist_ok=True)

        logger.info("Starting optimized parameter sweep (Excel already cached)")

        for pH in self.config.pH_list:
            pH_dir = results_dir / f"pH_{pH}"
            pH_dir.mkdir(exist_ok=True)

            # Determine sweep order
            if getattr(self.config, "enable_sweep_mode", False):
                V_steps = sorted(self.config.V_list, key=lambda v: abs(v))
            else:
                V_steps = list(self.config.V_list)

            previous_coverage = None

            for idx, V in enumerate(V_steps):
                # Update status if callback provided
                if status_callback:
                    status_callback(pH, V)
                V_dir = pH_dir / f"V_{V}"
                V_dir.mkdir(exist_ok=True)
                os.chdir(V_dir)

                try:
                    # Get data from CACHED processor (no file I/O!)
                    data_dict = self.excel_processor.get_data_for_conditions(pH, V)
                    
                    # Verify pH and V are correct
                    logger.info(f"Processing pH={pH}, V={V}")
                    logger.debug(f"  Data dict contains: pH={data_dict.get('pH')}, V={data_dict.get('V')}")
                    
                    # Ensure data_dict has the correct pH and V values
                    data_dict['pH'] = pH
                    data_dict['V'] = V

                    # Coverage propagation logic
                    if V == 0.0:
                        adsorbates = data_dict.get('adsorbates', [])
                        data_dict['activity'] = [0.0] * len(adsorbates)
                        data_dict['free_site_coverage'] = 1.0
                        logger.debug("Initial step V=0.0: zeroed adsorbates")
                    else:
                        if self.config.use_coverage_propagation and previous_coverage:
                            data_dict = self._apply_initial_coverage(data_dict, previous_coverage)
                            if '*' in previous_coverage:
                                data_dict['free_site_coverage'] = self._sanitize_value(previous_coverage.get('*', 1.0))
                            else:
                                data_dict['free_site_coverage'] = 1.0
                            logger.debug(f"Step V={V}: propagated coverage from previous step")
                        else:
                            data_dict['free_site_coverage'] = 1.0

                    # Enforce site balance
                    if self.ENFORCE_SITE_BALANCE and 'adsorbates' in data_dict and 'activity' in data_dict:
                        sanitized_acts, theta_free = self._renormalize_free_site(
                            data_dict['adsorbates'], data_dict['activity']
                        )
                        data_dict['activity'] = sanitized_acts
                        data_dict['free_site_coverage'] = theta_free
                    else:
                        data_dict['activity'] = [self._sanitize_value(a) for a in data_dict.get('activity', [])]
                        data_dict['free_site_coverage'] = self._sanitize_value(data_dict.get('free_site_coverage', 1.0))

                    # Calculate step time
                    time_per_step = self.config.time
                    if getattr(self.config, "enable_sweep_mode", False):
                        try:
                            time_per_step = self.config.calculate_step_time()
                        except Exception as e:
                            logger.warning(f"Failed to calculate step time: {e}")

                    sim_params = SimulationParameters(
                        temperature=self.config.temperature,
                        potential=V,  # Use the loop variable V
                        time=time_per_step,
                        abstol=self.config.abstol,
                        reltol=self.config.reltol,
                        pressure=data_dict.get('P', -1),
                        pH=pH,  # Use the loop variable pH
                        pre_exponential_factor=self.config.pre_exponential_factor,
                    )
                    
                    logger.debug(f"  SimParams: pH={sim_params.pH}, V={sim_params.potential}, T={sim_params.temperature}")

                    # Generate input and run
                    input_file = self.generator.generate_input_file(data_dict, sim_params)
                    
                    if self.config.executable_path:
                        start_time = time.perf_counter()
                        self.generator.run_simulation(input_file)
                        elapsed = time.perf_counter() - start_time
                        logger.info(f"pH={pH}, V={V} completed in {elapsed:.2f}s")

                        # Extract and save coverage
                        if getattr(self.config, "enable_sweep_mode", False):
                            final_cov = self._extract_final_coverage(V_dir)
                            if final_cov:
                                final_cov = self._sanitize_mapping(final_cov)
                                if self.ENFORCE_SITE_BALANCE:
                                    ads_list = data_dict.get('adsorbates', [])
                                    ads_vals = [final_cov.get(a, 0.0) for a in ads_list]
                                    ads_vals, theta_free = self._renormalize_free_site(ads_list, ads_vals)
                                    for a, v in zip(ads_list, ads_vals):
                                        final_cov[a] = v
                                    final_cov['*'] = theta_free
                                self.coverage_manager.save_coverage(pH, V, final_cov)
                                previous_coverage = final_cov
                    else:
                        logger.warning("Executable not set; skipping simulation")

                except Exception as e:
                    logger.error(f"Error at pH={pH}, V={V}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    os.chdir(base_dir)

        # Export coverage trajectory
        if getattr(self.config, "enable_sweep_mode", False):
            traj_file = results_dir / "coverage_trajectory.json"
            self.coverage_manager.export_coverage_trajectory(str(traj_file))

    def _apply_initial_coverage(self, data: Dict[str, Any], prev_coverage: Dict[str, float]) -> Dict[str, Any]:
        """Apply coverage from previous step as initial conditions."""
        data_copy = data.copy()
        if 'adsorbates' in data_copy and 'activity' in data_copy:
            new_activity = []
            for i, adsorbate in enumerate(data_copy['adsorbates']):
                raw_value = prev_coverage.get(adsorbate, data_copy['activity'][i])
                clean_value = self._sanitize_value(float(raw_value))
                new_activity.append(clean_value)
            data_copy['activity'] = new_activity
        return data_copy

    def _extract_final_coverage(self, simulation_dir: Path) -> Optional[Dict[str, float]]:
        """Extract final coverage from coverage.dat."""
        try:
            search_root = Path("run") / "range"
            coverage_files = list(search_root.rglob("coverage.dat"))

            if not coverage_files:
                logger.warning(f"No coverage.dat found in {search_root}")
                return None

            coverage_file = coverage_files[0]
            lines = coverage_file.read_text().strip().splitlines()
            
            if len(lines) < 2:
                return None

            headers = lines[0].split()
            last_values = lines[-1].split()

            n = min(len(headers), len(last_values))
            final_cov = {}
            
            for name, value in zip(headers[:n], last_values[:n]):
                try:
                    final_cov[name] = float(value)
                except ValueError:
                    continue

            return final_cov
            
        except Exception as e:
            logger.error(f"Error extracting coverage: {e}")
            return None