"""
Plotting module for microkinetic modeling results visualization.
Fixed all HTML entities, indentation errors, and improved functionality.
"""

import os
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import pandas as pd
import numpy as np
from matplotlib import rc, rcParams

logger = logging.getLogger(__name__)

class CoveragePlotter:
    """Handles plotting of coverage data from microkinetic simulations."""

    def __init__(self, base_directory: str = None):
        """
        Initialize plotter with base directory.

        Args:
            base_directory: Base directory containing simulation results
        """
        self.base_directory = Path(base_directory) if base_directory else Path.cwd()
        self._setup_plotting_style()

    def _setup_plotting_style(self) -> None:
        """Set up matplotlib plotting style."""
        rc('axes', linewidth=2)
        plt.rcParams.update({
            'font.size': 12,
            'font.weight': 'bold',
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold'
        })

    def read_coverage_data(self, pH: float, V: float) -> Dict[str, List[float]]:
        """
        Read coverage data from simulation results.
        Fixed path handling issues from original code.

        Args:
            pH: pH value
            V: Potential value

        Returns:
            Dictionary with species as keys and coverage lists as values
        """
        try:
            # Construct path to coverage data
            data_path = self.base_directory / f"pH_{pH}" / f"V_{V}"

            # Find run directory
            run_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("run")]
            if not run_dirs:
                logger.warning(f"No run directory found in {data_path}")
                return {}

            run_dir = run_dirs[0]  # Take first run directory
            coverage_file = run_dir / "range" / "coverage.dat"

            if not coverage_file.exists():
                logger.warning(f"Coverage file not found: {coverage_file}")
                return {}

            # Read coverage data
            with open(coverage_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                return {}

            # Parse header
            headers = lines[0].strip().split()
            coverage_data = {header: [] for header in headers}

            # Parse data lines
            # Parse data lines with negative value handling
            for line in lines[1:]:
                if line.strip():
                    values = list(map(float, line.strip().split()))
                    for i, header in enumerate(headers):
                        if i < len(values):
                            # Set negative coverage values to zero
                            coverage_value = max(0.0, values[i])
                            coverage_data[header].append(coverage_value)
                            if values[i] < 0:
                                logger.debug(f"Negative coverage found for {header}: {values[i]} -> set to 0.0")

            return coverage_data

        except Exception as e:
            logger.error(f"Error reading coverage data for pH={pH}, V={V}: {e}")
            return {}

    def get_final_coverages(self, pH_list: List[float], V_list: List[float]) -> Dict[float, Dict[float, Dict[str, float]]]:
        """
        Get final coverage values for all pH and V combinations.

        Args:
            pH_list: List of pH values
            V_list: List of potential values

        Returns:
            Nested dictionary: {pH: {V: {species: final_coverage}}}
        """
        all_coverages = {}

        for pH in pH_list:
            all_coverages[pH] = {}

            for V in V_list:
                coverage_data = self.read_coverage_data(pH, V)
                final_coverages = {}

                # Get final coverage for each species
                # Get final coverage for each species
                for species, values in coverage_data.items():
                    if values and '*' in species:  # Only adsorbates
                        # Ensure final coverage is non-negative
                        final_coverage = max(0.0, values[-1])
                        final_coverages[species] = final_coverage


                all_coverages[pH][V] = final_coverages

        return all_coverages

    def plot_coverage_vs_potential(self, pH_list: List[float], V_list: List[float], 
                                 save_plots: bool = True, show_plots: bool = True,
                                 output_dir: str = "plots") -> None:
        """
        Plot coverage vs potential for each pH.
        Fixed HTML entities and improved plotting logic.

        Args:
            pH_list: List of pH values to plot
            V_list: List of potential values
            save_plots: Whether to save plots
            show_plots: Whether to display plots
            output_dir: Directory to save plots
        """
        if save_plots:
            plots_dir = Path(output_dir)
            plots_dir.mkdir(exist_ok=True)

        # Get all coverage data
        all_coverages = self.get_final_coverages(pH_list, V_list)

        for pH in pH_list:
            if pH not in all_coverages:
                continue

            plt.figure(figsize=(10, 8))

            # Collect data for plotting
            species_data = {}
            for V in V_list:
                for species, coverage in all_coverages[pH].get(V, {}).items():
                    if species not in species_data:
                        species_data[species] = []
                    species_data[species].append(coverage)

            # Filter species with reasonable coverage values
            # Fixed HTML entities: &lt; -> <, &gt; -> >
            filtered_data = {}
            for species, coverages in species_data.items():
                if len(coverages) == len(V_list):  # Complete data
                    max_cov = max(coverages)
                    min_cov = min(coverages)
                    if max_cov <= 1 and min_cov >= 1e-20:  # Reasonable range
                        filtered_data[species] = coverages

            # Plot each species
            for species, coverages in filtered_data.items():
                label = self._format_species_name(species)
                plt.plot(V_list, coverages, label=label, linewidth=2, marker='o')

            # Formatting
            plt.xlabel('Potential (V)', fontsize=16, fontweight='bold')
            plt.ylabel('Coverage', fontsize=16, fontweight='bold')
            plt.title(f'Coverage vs Potential (pH = {pH})', fontsize=20, fontweight='bold')

            # Legend
            if filtered_data:
                legend_properties = {'weight': 'bold'}
                leg = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                               prop=legend_properties, fontsize=12)
                leg.get_frame().set_edgecolor('black')
                leg.get_frame().set_linewidth(2.0)

            plt.xticks(fontweight='bold', fontsize=14)
            plt.yticks(fontweight='bold', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_plots:
                plt.savefig(plots_dir / f'coverage_pH_{pH}.png', dpi=300, bbox_inches='tight')
                logger.info(f"Saved plot: coverage_pH_{pH}.png")

            if show_plots:
                plt.show()
            else:
                plt.close()

    def _format_species_name(self, species: str) -> str:
        """Format species name for plotting (convert to subscripts)."""
        # Simple subscript conversion for common species
        normal = "0123456789"
        sub_s = "₀₁₂₃₄₅₆₇₈₉"
        trans_table = str.maketrans(normal, sub_s)

        formatted = species.translate(trans_table)

        # Move * to the beginning if present
        if '*' in formatted and not formatted.startswith('*'):
            formatted = formatted.replace('*', '')
            formatted = '*' + formatted

        return formatted

    def read_derivatives_data(self, pH: float, V: float) -> Dict[str, float]:
        """Read rates from derivatives.dat."""
        try:
            data_path = self.base_directory / f"pH_{pH}" / f"V_{V}"
            run_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("run")]
            if not run_dirs: return {}
            
            run_dir = run_dirs[0]
            deriv_file = run_dir / "range" / "derivatives.dat"
            
            if not deriv_file.exists(): return {}
            
            with open(deriv_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2: return {}
            
            headers = lines[0].strip().split()
            values = lines[1].split() # Take first data line (usually only one for final state)
            
            # Map headers to values
            return {h: float(v) for h, v in zip(headers, values) if v.strip()}
            
        except Exception:
            return {}

    def plot_current_density(self, pH_list: List[float], V_list: List[float], 
                           output_dir: str = "plots", 
                           site_density: float = 2.94e-5,
                           target_species: List[str] = None,
                           species_electrons: Dict[str, int] = None) -> None:
        """Plot current density vs potential."""
        # Validation checks
        if not target_species:
            logger.info("No target species specified for current density plot. Skipping.")
            return

        if not species_electrons:
             # Try default if empty
             logger.warning("No species electron map provided. Using defaults.")
             species_electrons = {'CH3OH': 6, 'CH4': 8, 'CO': 2, 'HCOOH': 2, 'H2': 2, 'CH2O': 4}
             
        # Filter target species that are actually in the electron map
        valid_species = [s for s in target_species if s in species_electrons]
        if not valid_species:
            logger.warning("None of the target species have defined electron counts. Skipping current density plot.")
            return

        F = 96485
        
        # Colors for pH
        pH_colors_list = ['crimson', 'mediumseagreen', 'dodgerblue', 'darkorange', 'forestgreen', 'slateblue', 'purple', 'black']
        pH_colors = {f"pH {p}": pH_colors_list[i % len(pH_colors_list)] for i, p in enumerate(pH_list)}
        
        # Markers and Linestyles
        markers_list = ['o', 'x', 's', '^', 'D', 'v', 'p', '*']
        linestyles_list = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 2)), '-', '--']
        
        markers = {s: markers_list[i % len(markers_list)] for i, s in enumerate(target_species)}
        linestyles = {s: linestyles_list[i % len(linestyles_list)] for i, s in enumerate(target_species)}

        # Pretty LaTeX-style labels map
        species_label_map = {
             'CH3OH': 'CH$_3$OH', 'CH4': 'CH$_4$', 'CO': 'CO', 'HCOOH': 'HCOOH', 
             'H2': 'H$_2$', 'CH2O': 'CH$_2$O', 'C2H4': 'C$_2$H$_4$', 'C2H5OH': 'C$_2$H$_5$OH'
        }

        plt.figure(figsize=(9, 6))
        
        has_data = False
        
        # Collect data first
        data_by_pH = {}
        for pH in pH_list:
            pH_label = f"pH {pH}"
            data_by_pH[pH_label] = []
            for V in V_list:
                rates = self.read_derivatives_data(pH, V)
                if rates:
                    data_by_pH[pH_label].append((V, rates))
        
        # Plotting loop
        # Sort manually by numeric pH value
        sorted_items = sorted(data_by_pH.items(), key=lambda x: float(x[0].split()[1]))
        
        for pH_label, data in sorted_items:
            data = sorted(data, key=lambda x: x[0])  # Sort by potential
            potentials = [entry[0] for entry in data]

            for species in target_species:
                if species not in species_electrons: continue
                
                rates = [entry[1].get(species, 1e-40) for entry in data]
                
                current_density = np.array([
                    rate * site_density * species_electrons[species] * F * 0.1 for rate in rates
                ])
                
                mask = current_density >= 1e-10
                
                if not np.any(mask):
                    continue

                label = fr'{species_label_map.get(species, species)}, {pH_label}'
                
                # Get color for pH
                color = pH_colors.get(pH_label, 'black')
                
                plt.plot(np.array(potentials)[mask], current_density[mask],
                         label=label,
                         color=color,
                         marker=markers.get(species, 'o'),
                         linestyle=linestyles.get(species, '-'),
                         linewidth=2,
                         markersize=8) # Slightly smaller than 12
                has_data = True

        if has_data:
            plt.xlabel('Potential U vs RHE (V)', fontsize=18)
            plt.ylabel('j (mA/cm²)', fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.yscale('log')  # Set y-axis to logarithmic scale
            
            # Bold axis lines
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_linewidth(2)

            plt.legend(fontsize=12, prop={'weight': 'bold'}, ncol=2, loc='upper right')
            plt.grid(False)
            plt.tight_layout()
            
            out_path = Path(output_dir) / "current_density.png"
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved current density plot: {out_path}")
        else:
            logger.warning("No valid current density data found to plot")
        
        plt.close()

    def plot_selectivity(self, pH_list: List[float], V_list: List[float], 
                       output_dir: str = "plots", 
                       target_species: List[str] = None) -> None:
        """Plot selectivity bar charts."""
        if target_species is None:
            target_species = ['CH3OH', 'CH4', 'CO', 'HCOOH', 'H2', 'CH2O']
            
        colors_map = {
            'CH3OH': 'tab:blue', 'CH4': 'tab:orange', 'CO': 'tab:green',
            'HCOOH': 'tab:red', 'H2': 'tab:purple', 'CH2O': 'tab:brown'
        }
        
        # One plot per pH
        for pH in pH_list:
            potentials = []
            selectivities = {s: [] for s in target_species}
            
            has_data_ph = False
            
            # Sort V_list
            sorted_V = sorted(V_list)
            
            for V in sorted_V:
                rates = self.read_derivatives_data(pH, V)
                if not rates: continue
                
                total_rate = sum(rates.get(s, 0.0) for s in target_species)
                if total_rate <= 0: continue
                
                potentials.append(f"{V:.2f}")
                has_data_ph = True
                
                for s in target_species:
                    val = rates.get(s, 0.0) / total_rate
                    selectivities[s].append(val)
            
            if not has_data_ph: continue
            
            # Plot
            plt.figure(figsize=(10, 6))
            x = np.arange(len(potentials))
            bottom = np.zeros(len(x))
            
            for species in target_species:
                if not selectivities[species]: continue
                
                label = self._format_species_name(species)
                plt.bar(x, selectivities[species], bottom=bottom, 
                       label=label, color=colors_map.get(species, 'gray'), width=0.6)
                bottom += np.array(selectivities[species])
                
            plt.xlabel('Potential (V)', fontsize=16, fontweight='bold')
            plt.ylabel('Selectivity', fontsize=16, fontweight='bold')
            plt.title(f'Selectivity at pH {pH}', fontsize=18, fontweight='bold')
            plt.xticks(x, potentials, rotation=45, ha='right', fontsize=12, fontweight='bold')
            plt.yticks(fontsize=12, fontweight='bold')
            plt.ylim(0, 1.05)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
            plt.grid(True, axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            out_path = Path(output_dir) / f"selectivity_pH_{pH}.png"
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved selectivity plot: {out_path}")
            plt.close()

    def create_coverage_summary_table(self, pH_list: List[float], V_list: List[float], 
                                    save_csv: bool = True, output_path: str = "coverage_summary.csv") -> pd.DataFrame:
        """
        Create a summary table of final coverages.

        Args:
            pH_list: List of pH values
            V_list: List of potential values
            save_csv: Whether to save as CSV
            output_path: Path for CSV file

        Returns:
            DataFrame with coverage summary
        """
        all_coverages = self.get_final_coverages(pH_list, V_list)

        # Collect all unique species
        all_species = set()
        for pH_data in all_coverages.values():
            for V_data in pH_data.values():
                all_species.update(V_data.keys())

        # Create summary data
        summary_data = []
        for pH in pH_list:
            for V in V_list:
                row = {'pH': pH, 'V': V}
                for species in sorted(all_species):
                    coverage = all_coverages.get(pH, {}).get(V, {}).get(species, 0.0)
                    row[species] = coverage
                summary_data.append(row)

        df = pd.DataFrame(summary_data)

        if save_csv:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved coverage summary: {output_path}")

        return df

def create_plots(pH_list: List[float], V_list: List[float], 
                base_directory: str = None, save_plots: bool = True, output_dir: str = None, **kwargs) -> None:
    """
    Convenience function to create all plots.
    This replaces the original plot.py functionality with proper structure.

    Args:
        pH_list: List of pH values
        V_list: List of potential values  
        base_directory: Base directory containing results
        save_plots: Whether to save plots
        output_dir: Optional directory to save plots into
    """
    plotter = CoveragePlotter(base_directory)
    
    # Use provided output_dir or default to "plots" in current dir (legacy behavior)
    # OR better, default to base_directory/plots if base_directory is provided?
    # For backward compatibility, let's stick to plotter defaults unless overridden.
    
    if output_dir is None:
        plot_output = "plots"
    else:
        plot_output = output_dir

    # 1. Coverage vs Potential
    try:
        plotter.plot_coverage_vs_potential(pH_list, V_list, save_plots=save_plots, output_dir=plot_output)
    except Exception as e:
        logger.warning(f"Failed to generate coverage plots: {e}")

    # 2. Current Density
    try:
        plotter.plot_current_density(pH_list, V_list, output_dir=plot_output, site_density=kwargs.get('site_density', 2.94e-5),
                                    target_species=kwargs.get('target_species', None),
                                    species_electrons=kwargs.get('species_electrons', None))
    except Exception as e:
        logger.warning(f"Failed to generate current density plots: {e}")

    # 3. Selectivity
    try:
        plotter.plot_selectivity(pH_list, V_list, output_dir=plot_output, target_species=kwargs.get('target_species', None))
    except Exception as e:
        logger.warning(f"Failed to generate selectivity plots: {e}")

    # 4. Summary Table
    try:
        plotter.create_coverage_summary_table(pH_list, V_list)
    except Exception as e:
        logger.warning(f"Failed to generate summary table: {e}")

    logger.info("Plotting process completed (check logs for specific warnings)")
