"""
Optimized main application that reads Excel once at startup.
All subsequent operations work with cached data in memory.
"""

import os
import sys
import logging
from pathlib import Path
import yaml
import json
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

# Import modules
from config import SolverSettings, load_config
from data_parser import CachedExcelDataProcessor
from simulation import OptimizedSimulationRunner
from plotting import CoveragePlotter, create_plots

logger = logging.getLogger(__name__)


class OptimizedMicrokineticModeling:
    """
    Optimized application that loads Excel once and caches all data.
    Significantly faster for parameter sweeps.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize application with configuration.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = load_config(config_path)
        self.validate_setup()
        
        # Load Excel data ONCE at initialization
        logger.info(f"Loading and caching Excel data from {self.config.input_excel_path}...")
        self.excel_processor = CachedExcelDataProcessor(self.config.input_excel_path)
        logger.info("✅ Excel data cached successfully - no further file reads needed!")

    def validate_setup(self) -> None:
        """Validate configuration and setup."""
        errors = self.config.validate()
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            raise ValueError("Invalid configuration")

        logger.info("Configuration validated successfully")
        logger.info(f"pH range: {self.config.pH_list}")
        logger.info(f"V range: {self.config.V_list}")
        logger.info(f"Temperature: {self.config.temperature} K")

    def run_simulations(self, status_callback=None) -> None:
        """Run all simulations using cached Excel data."""
        logger.info("Starting optimized simulation parameter sweep")
        
        # Initialize runner with cached processor
        runner = OptimizedSimulationRunner(self.config, self.excel_processor)
        
        # Run parameter sweep (no Excel I/O needed)
        runner.run_parameter_sweep(status_callback=status_callback)
        
        logger.info("Simulation parameter sweep completed")

    def create_plots(self) -> None:
        """Create plots from simulation results."""
        if not self.config.create_plots:
            logger.info("Plotting disabled in configuration")
            return

        logger.info("Creating plots from simulation results")

        # Explicitly pass output directory to plotting to ensure it goes to the right place
        # The plotting module defaults to "plots", let's make it explicitly use a plots subdir of results
        plots_output_dir = str(Path(self.config.output_base_dir) / "plots")
        
        create_plots(
            pH_list=self.config.pH_list,
            V_list=self.config.V_list,
            base_directory=self.config.output_base_dir,
            save_plots=True,
            output_dir=plots_output_dir,
            site_density=getattr(self.config, 'site_density', 2.94e-5),
            target_species=getattr(self.config, 'target_species', None),
            species_electrons=getattr(self.config, 'species_electrons', None)
        )

        logger.info("Plotting completed")

    def run_full_workflow(self, status_callback=None) -> None:
        """Run the complete workflow: simulations + plotting."""
        try:
            self.run_simulations(status_callback=status_callback)
            self.create_plots()
            logger.info("✅ Full workflow completed successfully")

        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}")
            raise

    def export_config(self, output_path: str) -> None:
        """Export current configuration to file."""
        if output_path.endswith('.yaml') or output_path.endswith('.yml'):
            self.config.to_yaml(output_path)
        elif output_path.endswith('.json'):
            self.config.to_json(output_path)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")

        logger.info(f"Configuration exported to: {output_path}")

    def benchmark_performance(self) -> None:
        """Run a performance benchmark comparing data access times."""
        import time
        
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE BENCHMARK")
        logger.info("="*60)
        
        # Test cached access speed
        pH = self.config.pH_list[0]
        V = self.config.V_list[0]
        
        start = time.perf_counter()
        for _ in range(100):
            data = self.excel_processor.get_data_for_conditions(pH, V)
        elapsed = time.perf_counter() - start
        
        logger.info(f"\n✅ Cached data access (100 iterations):")
        logger.info(f"   Total time: {elapsed:.4f} seconds")
        logger.info(f"   Per iteration: {elapsed/100*1000:.2f} ms")
        logger.info(f"\n💡 Excel file opened: ZERO times")
        logger.info(f"   All data served from memory cache")
        logger.info("="*60 + "\n")


def create_example_config() -> None:
    """Create an example configuration file."""
    config = SolverSettings()
    config.pH_list = [7, 10, 13]
    config.V_list = [0, -0.2, -0.4, -0.6, -0.8, -1.0]
    config.executable_path = "/path/to/your/mkmcxx.exe"
    config.input_excel_path = "input.xlsx"

    config.to_yaml("example_config.yaml")
    config.to_json("example_config.json")

    print("Created example configuration files:")
    print("  - example_config.yaml")  
    print("  - example_config.json")


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description='Optimized Microkinetic Modeling Workflow (Excel cached at startup)'
    )

    parser.add_argument('--config', '-c', type=str, 
                       help='Path to configuration file')
    parser.add_argument('--simulations-only', action='store_true',
                       help='Run only simulations (no plotting)')
    parser.add_argument('--plots-only', action='store_true', 
                       help='Create only plots (no simulations)')
    parser.add_argument('--create-example-config', action='store_true',
                       help='Create example configuration files')
    parser.add_argument('--export-config', type=str,
                       help='Export current config to specified file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    
    parser.add_argument('--sweep-mode', action='store_true', 
                       help='Enable sweep mode with coverage propagation')
    parser.add_argument('--sweep-rate', type=float, default=0.1, 
                       help='Sweep rate in V/s (default: 0.1)')
    parser.add_argument('--no-coverage-propagation', action='store_true', 
                       help='Disable coverage propagation in sweep mode')

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Handle special commands
    if args.create_example_config:
        create_example_config()
        return

    try:
        # Initialize optimized application (Excel loaded once here)
        logger.info("Initializing optimized microkinetic modeling application...")
        app = OptimizedMicrokineticModeling(args.config)

        # Flatten V_list if nested
        if any(isinstance(v, list) for v in app.config.V_list):
            app.config.V_list = [float(x) for sub in app.config.V_list for x in sub]
            logger.warning(f"Flattened nested V_list → {app.config.V_list}")

        # Override sweep mode settings from command line
        if args.sweep_mode:
            app.config.enable_sweep_mode = True
            app.config.sweep_rate = args.sweep_rate
            app.config.use_coverage_propagation = not args.no_coverage_propagation
            logger.info(f"Sweep mode enabled: {args.sweep_rate} V/s, "
                       f"coverage propagation: {not args.no_coverage_propagation}")

        # Handle config export
        if args.export_config:
            app.export_config(args.export_config)
            return

        # Run benchmark if requested
        if args.benchmark:
            app.benchmark_performance()
            return

        # Run workflow based on arguments
        if args.simulations_only:
            app.run_simulations()
        elif args.plots_only:
            app.create_plots()
        else:
            app.run_full_workflow()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
