#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preparation for ML analysis of STM MTRX data

This file contains functions for preparing data
for training and prediction.
    
@author: Steven R. Schofield 

Created May 2025

"""

# ============================================================================
# Module dependencies
# ============================================================================

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path



# ============================================================================
# Basic file io
# ============================================================================
def find_all_castep_files(root_dir):
    """
    Recursively finds all .castep files under the given root directory.
    
    Parameters:
        root_dir (str or Path): The directory to search from.
        
    Returns:
        List[Path]: List of Path objects pointing to .castep files.
    """
    root = Path(root_dir)
    return list(root.rglob("*.castep"))


# ============================================================================
#  General information
# ============================================================================

def print_file_info(castep_path):
    """
    Prints a clear heading with filename and full path.
    """
    path = Path(castep_path)
    filename = path.name
    parent_path = path.parent

    # Build a consistent-width header
    header_text = f" FILE: {filename} "
    path_text   = f" PATH: {parent_path} "
    width = max(len(header_text), len(path_text)) + 4

    print("\n" + "=" * width)
    print(header_text.center(width))
    print(path_text.center(width))
    print("=" * width + "\n")

def extract_warnings(castep_path, verbose=True):
    """
    Extracts and prints WARNING blocks from a .castep file.
    
    Parameters:
        castep_path (str or Path): Path to the .castep file.
        verbose (bool): If True, print full blocks until the next blank line.
                        If False, only print the matching WARNING line.
    """
    path = Path(castep_path)
    filename = path.name
    parent_path = path.parent

    with open(castep_path, 'r') as f:
        lines = f.readlines()

    in_warning = False
    current_warning = []
    any_warning_found = False

    for i, line in enumerate(lines):
        if "warning" in line.lower():
            if not any_warning_found:
                print(f"\n===== WARNINGS in: {filename} =====")
                print(f"      full path: {parent_path}\n")
                any_warning_found = True

            if not verbose:
                print(f"Line {i+1}: {line.strip()}")
                continue

            # Verbose mode: collect block
            if current_warning:
                print("".join(current_warning).rstrip())
                print("-" * 40)
            in_warning = True
            current_warning = [line]
        elif in_warning:
            if line.strip() == '':
                print("".join(current_warning).rstrip())
                print("-" * 40)
                in_warning = False
                current_warning = []
            else:
                current_warning.append(line)

    if in_warning and current_warning and verbose:
        print("".join(current_warning).rstrip())
        print("-" * 40)

    if not any_warning_found:
        print(f"No warnings found in: {filename}")
        print(f"  full path: {parent_path}")


# ============================================================================
#  Energy and structure information
# ============================================================================
def extract_LBFGS_energies(castep_path):
    """
    Extracts iteration numbers and enthalpy values from LBFGS optimization steps.
    Returns a list of tuples: (iteration_number, enthalpy_in_eV)
    
    Matches lines like:
    LBFGS: finished iteration     0 with enthalpy= -8.36353629E+003 eV
    """
    results = []

    pattern = re.compile(
        r'LBFGS: finished iteration\s+(\d+)\s+with enthalpy=\s*([-+]?\d*\.\d+E[+-]?\d+|\d+)'
    )

    with open(castep_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                iteration = int(match.group(1))
                enthalpy = float(match.group(2))
                results.append((iteration, enthalpy))

    return results

def extract_LBFGS_final_enthalpy(castep_path):
    """
    Extracts the final enthalpy value from a line like:
    'LBFGS: Final Enthalpy     = -8.36355887E+003 eV'
    
    Ensures that exactly one such line exists in the file.
    
    Returns:
        float: The enthalpy value in eV.
    
    Raises:
        ValueError: if no match or multiple matches are found.
    """
    pattern = re.compile(r'LBFGS: Final Enthalpy\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][+-]?\d+))')
    matches = []

    with open(castep_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                matches.append(float(match.group(1)))

    if len(matches) == 0:
        raise ValueError(f"No final enthalpy line found in file: {castep_path}")
    if len(matches) > 1:
        raise ValueError(f"Multiple final enthalpy lines found in file: {castep_path}")

    return matches[0]

def extract_lattice_parameters(castep_path):
    with open(castep_path, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines) - 1, 0, -1):
        if "Lattice parameters" in lines[i]:
            # Should be followed by 3 lines like: a = 5.43  alpha = 90.0
            a_line = lines[i+1].strip().split()
            b_line = lines[i+2].strip().split()
            c_line = lines[i+3].strip().split()

            a = float(a_line[2])
            alpha = float(a_line[5])
            b = float(b_line[2])
            beta = float(b_line[5])
            c = float(c_line[2])
            gamma = float(c_line[5])
            return {'a': a, 'b': b, 'c': c, 'alpha': alpha, 'beta': beta, 'gamma': gamma}
    return None


def extract_lattice_parameters(castep_path, a0=3.8668346, vac=15.0):
    ax, ay, az = 'err', 'err', 'err'
    nx, ny, nz = 'err', 'err', 'err'
    with open(castep_path, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines) - 1, 0, -1):
        if "Lattice parameters" in lines[i]:
            # Should be followed by 3 lines like: a = 5.43  alpha = 90.0
            a_line = lines[i+1].strip().split()
            b_line = lines[i+2].strip().split()
            c_line = lines[i+3].strip().split()

            ax = float(a_line[2])
            alpha = float(a_line[5])
            ay = float(b_line[2])
            beta = float(b_line[5])
            az = float(c_line[2])
            gamma = float(c_line[5])

            try:
                nx_temp = ax / a0
                ny_temp = ay / a0
                nz_temp = (az - vac) / (a0 * np.sqrt(2) / 4 )

                # Check each dimension separately for integer status after rounding to 6 decimal places
                nx = round(nx_temp) if round(nx_temp, 6).is_integer() else 'err'
                ny = round(ny_temp) if round(ny_temp, 6).is_integer() else 'err'
                nz = round(nz_temp) if round(nz_temp, 6).is_integer() else 'err'

            except Exception as e:
                ax, ay, az = 'err', 'err', 'err'
                nx, ny, nz = 'err', 'err', 'err'

    return {'ax': ax, 'ay': ay, 'az': az, 'nx': nx, 'ny': ny, 'nz': nz, 'alpha': alpha, 'beta': beta, 'gamma': gamma}

# ============================================================================
#  Plotting functions
# ============================================================================

def plot_energy_vs_iteration(data, ylabel="Energy (eV)", title="Energy Convergence", figsize=(6, 4)):
    if not data:
        print("No data to plot.")
        return

    iterations, energies = zip(*data)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(iterations, energies, marker='o', linestyle='-')
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    # ――― key line ―――
    ax.ticklabel_format(style='plain', axis='y', useOffset=False)

    plt.tight_layout()
    plt.show()
