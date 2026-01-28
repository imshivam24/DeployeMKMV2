"""
Utility functions and classes for microkinetic modeling.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class FileManager:
    """Manages file operations for the microkinetic modeling workflow."""

    @staticmethod
    def create_directory_structure(base_dir: str, pH_list: List[float], V_list: List[float]) -> None:
        """
        Create directory structure for simulation results.

        Args:
            base_dir: Base directory path
            pH_list: List of pH values
            V_list: List of potential values
        """
        base_path = Path(base_dir)
        base_path.mkdir(exist_ok=True)

        for pH in pH_list:
            pH_dir = base_path / f"pH_{pH}"
            pH_dir.mkdir(exist_ok=True)

            for V in V_list:
                V_dir = pH_dir / f"V_{V}"
                V_dir.mkdir(exist_ok=True)

        logger.info(f"Created directory structure in {base_dir}")

    @staticmethod
    def backup_results(source_dir: str, backup_name: str = None) -> str:
        """
        Create a backup of simulation results.

        Args:
            source_dir: Directory to backup
            backup_name: Name for backup (optional)

        Returns:
            Path to backup directory
        """
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"

        source_path = Path(source_dir)
        backup_path = source_path.parent / backup_name

        if source_path.exists():
            shutil.copytree(source_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return str(backup_path)
        else:
            logger.warning(f"Source directory not found: {source_dir}")
            return ""

class DataValidator:
    """Validates simulation data and results."""

    @staticmethod
    def validate_reaction_data(reactions: List[str], Ea: List[float], Eb: List[float]) -> List[str]:
        """
        Validate reaction data for consistency.

        Args:
            reactions: List of reaction strings
            Ea: Forward activation energies
            Eb: Backward activation energies

        Returns:
            List of validation errors
        """
        errors = []

        if len(reactions) != len(Ea):
            errors.append(f"Mismatch: {len(reactions)} reactions but {len(Ea)} forward barriers")

        if len(reactions) != len(Eb):
            errors.append(f"Mismatch: {len(reactions)} reactions but {len(Eb)} backward barriers")

        # Check for valid reaction format
        for i, rxn in enumerate(reactions):
            if "→" not in rxn and "->" not in rxn:
                errors.append(f"Invalid reaction format at index {i}: {rxn}")

        # Check for reasonable energy values
        for i, ea in enumerate(Ea):
            if ea < 0:
                errors.append(f"Negative forward barrier at index {i}: {ea}")
            if ea > 1e6:
                errors.append(f"Unreasonably high forward barrier at index {i}: {ea}")

        for i, eb in enumerate(Eb):
            if eb < 0:
                errors.append(f"Negative backward barrier at index {i}: {eb}")
            if eb > 1e6:
                errors.append(f"Unreasonably high backward barrier at index {i}: {eb}")

        return errors

class PerformanceMonitor:
    """Monitors and reports on simulation performance."""

    def __init__(self):
        self.start_times = {}
        self.durations = {}

    def start_timer(self, task_name: str) -> None:
        """Start timing a task."""
        self.start_times[task_name] = datetime.now()

    def end_timer(self, task_name: str) -> float:
        """End timing a task and return duration in seconds."""
        if task_name not in self.start_times:
            logger.warning(f"Timer for {task_name} was not started")
            return 0.0

        duration = (datetime.now() - self.start_times[task_name]).total_seconds()
        self.durations[task_name] = duration

        logger.info(f"{task_name} completed in {duration:.2f} seconds")
        return duration

    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of all timed tasks."""
        return self.durations.copy()

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator
        denominator: Denominator  
        default: Default value if division by zero

    Returns:
        Result of division or default value
    """
    return numerator / denominator if denominator != 0 else default

def create_summary_report(results_dir: str, output_file: str = "summary_report.txt") -> None:
    """
    Create a summary report of simulation results.

    Args:
        results_dir: Directory containing simulation results
        output_file: Output file for report
    """
    results_path = Path(results_dir)

    with open(output_file, 'w') as f:
        f.write("Microkinetic Modeling - Summary Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results directory: {results_dir}\n\n")

        if not results_path.exists():
            f.write("Results directory not found!\n")
            return

        # Count simulations
        pH_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('pH_')]

        f.write(f"Number of pH conditions: {len(pH_dirs)}\n")

        total_simulations = 0
        for pH_dir in pH_dirs:
            V_dirs = [d for d in pH_dir.iterdir() if d.is_dir() and d.name.startswith('V_')]
            total_simulations += len(V_dirs)

        f.write(f"Total simulations: {total_simulations}\n\n")

    logger.info(f"Summary report created: {output_file}")
