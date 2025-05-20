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
import math
from ase import Atoms
import nglview as nv        # pip install nglview

from IPython.display import display, Image as StaticImage
import time

# ============================================================================
# Basic file io
# ============================================================================
def find_all_files_by_extension(root_dir, extension=".castep"):
    """
    Recursively finds all files with the given extension under the specified directory.
    
    Parameters:
        root_dir (str or Path): Directory to search from.
        extension (str): File extension to match, including the dot (e.g. ".castep", ".xyz").
    
    Returns:
        List[Path]: List of matching Path objects.
    """
    root = Path(root_dir)
    if not extension.startswith("."):
        extension = "." + extension
    return list(root.rglob(f"*{extension}"))


# ============================================================================
#  General information
# ============================================================================

def print_filename(castep_path):
    """
    Prints a clear heading with filename and full path.
    """
    path = Path(castep_path)
    filename = path.name
    parent_path = path.parent
    full_path = parent_path / filename

    # Build a consistent-width header
    header_text = f" FILE: {filename} "
    path_text   = f" PATH: {full_path} "
    width = max(len(header_text), len(path_text)) + 4

    print("\n" + "=" * width)
    print(header_text.center(width))
    print(path_text.center(width))
    print("=" * width + "\n")


def get_warnings(castep_path, verbose=True):
    """
    Extracts WARNING blocks from a .castep file and returns them as text.
    
    Parameters:
        castep_path (str or Path): Path to the .castep file.
        verbose (bool): If True, include full blocks until the next blank line.
                        If False, include only the matching WARNING line.
    
    Returns:
        str: The formatted warning output (or a 'no warnings' message).
    """
    path = Path(castep_path)
    filename = path.name
    parent_path = path.parent
    full_path = parent_path / filename

    with open(castep_path, 'r') as f:
        lines = f.readlines()

    output_lines = []
    in_warning = False
    current_warning = []
    any_warning_found = False

    for i, line in enumerate(lines):
        if "warning" in line.lower():
            if not any_warning_found:
                output_lines.append(f"\n===== WARNINGS in: {filename} =====")
                output_lines.append(f"      full path: {full_path}\n")
                any_warning_found = True

            if not verbose:
                output_lines.append(f"Line {i+1}: {line.strip()}")
                continue

            # Verbose mode: start a new block
            if current_warning:
                # flush previous block
                output_lines.append("".join(current_warning).rstrip())
                output_lines.append("-" * 40)
            in_warning = True
            current_warning = [line]

        elif in_warning:
            if line.strip() == "":
                # end of block
                output_lines.append("".join(current_warning).rstrip())
                output_lines.append("-" * 40)
                in_warning = False
                current_warning = []
            else:
                current_warning.append(line)

    # flush if file ended while still in a warning block
    if in_warning and current_warning and verbose:
        output_lines.append("".join(current_warning).rstrip())
        output_lines.append("-" * 40)

    if not any_warning_found:
        output_lines.append(f"No warnings found in: {filename}")
        output_lines.append(f"  full path: {full_path}")

    return "\n".join(output_lines)


def get_calculation_parameters(castep_path):
    """
    Extracts key calculation parameters from a CASTEP .castep file.
    Returns a dictionary of parameters and values (as floats, ints, or strings),
    including MP grid size, k-point offset, and number of k-points.
    """
    keys_of_interest = {
        'plane wave basis set cut-off',
        'finite basis set correction',
        'number of  electrons',
        'net charge of system',
        'net spin   of system',
        'number of  up  spins',
        'number of down spins',
        'treating system as spin-polarized',
        'number of bands',
        'total energy / atom convergence tol.',
        'eigen-energy convergence tolerance',
        'max force / atom convergence tol.',
        'convergence tolerance window',
        'smearing scheme',
        'smearing width',
        'Fermi energy convergence tolerance',
        'periodic dipole correction'
    }

    results = {}

    with open(castep_path, 'r') as f:
        for line in f:
            # Handle normal key:value pairs
            for key in keys_of_interest:
                if key in line:
                    parts = line.split(":", 1)
                    if len(parts) < 2:
                        continue
                    value = parts[1].strip()
                    try:
                        results[key] = float(value)
                    except ValueError:
                        try:
                            results[key] = int(value)
                        except ValueError:
                            results[key] = value
                    break  # avoid matching multiple keys per line

            # MP grid size
            if "MP grid size for SCF calculation is" in line:
                match = re.findall(r'\d+', line)
                if len(match) == 3:
                    results['kx'] = int(match[0])
                    results['ky'] = int(match[1])
                    results['kz'] = int(match[2])

            # Offset
            if "with an offset of" in line:
                match = re.findall(r'[-+]?\d*\.\d+', line)
                if len(match) == 3:
                    results['k_offset'] = tuple(float(x) for x in match)

            # Number of k-points
            if "Number of kpoints used" in line:
                match = re.search(r'=\s*(\d+)', line)
                if match:
                    results['n_kpoints'] = int(match.group(1))

    return results


def collect_summary_table(data_path):
    """
    Scans all .castep files under data_path and builds a summary table with:
    - filename (no extension)
    - relative path to data_path
    - nx, ny, nz
    - kx, ky, kz
    - cut-off energy
    - net charge and net spin
    - final enthalpy
    Returns: pandas DataFrame
    """
    job_path = Path(data_path).resolve()
    castep_files = find_all_files_by_extension(job_path, extension=".castep")
    
    summary = []

    for castep_path in castep_files:
        castep_path = Path(castep_path).resolve()
        rel_path = castep_path.parent.relative_to(job_path)
        filename = castep_path.stem

        # Get data
        try:
            cell = extract_lattice_parameters(castep_path)
            general = extract_summary_parameters(castep_path)
            enthalpy = extract_LBFGS_final_enthalpy(castep_path)
        except Exception as e:
            print(f"Skipping {castep_path} due to error: {e}")
            continue

        # Cell params
        nx = cell.get("nx", None)
        ny = cell.get("ny", None)
        nz = cell.get("nz", None)

        # MP grid params
        kx = general.get("kx", None)
        ky = general.get("ky", None)
        kz = general.get("kz", None)

        # Other general params
        cut = general.get("plane wave basis set cut-off", None)
        charge = general.get("net charge of system", None)
        spin = general.get("net spin   of system", None)

        summary.append({
            "File": filename,
            "RelPath": str(rel_path),
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "kx": kx,
            "ky": ky,
            "kz": kz,
            "Cut-off (eV)": cut,
            "Net Charge": charge,
            "Net Spin": spin,
            "Final Enthalpy (eV)": enthalpy
        })

    return pd.DataFrame(summary)


# ============================================================================
#  Energy information
# ============================================================================
def get_LBFGS_energies(castep_path):
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

def get_LBFGS_final_enthalpy(castep_path):
    """
    Extracts the final enthalpy value from a line like:
    'LBFGS: Final Enthalpy     = -8.36355887E+003 eV'
    
    Returns:
        float or str: Final enthalpy value in eV, or "err" if not found or ambiguous.
    """
    pattern = re.compile(r'LBFGS: Final Enthalpy\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][+-]?\d+))')
    matches = []

    try:
        with open(castep_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    matches.append(float(match.group(1)))

        if len(matches) != 1:
            return float('nan')  # or return None

        return matches[0]

    except Exception as e:
        return float('nan')  # or return None





# ============================================================================
#  Structure information
# ============================================================================

# def get_lattice_parameters(castep_path):
#     with open(castep_path, 'r') as f:
#         lines = f.readlines()

#     for i in range(len(lines) - 1, 0, -1):
#         if "Lattice parameters" in lines[i]:
#             # Should be followed by 3 lines like: a = 5.43  alpha = 90.0
#             a_line = lines[i+1].strip().split()
#             b_line = lines[i+2].strip().split()
#             c_line = lines[i+3].strip().split()

#             a = float(a_line[2])
#             alpha = float(a_line[5])
#             b = float(b_line[2])
#             beta = float(b_line[5])
#             c = float(c_line[2])
#             gamma = float(c_line[5])
#             return {'a': a, 'b': b, 'c': c, 'alpha': alpha, 'beta': beta, 'gamma': gamma}
#     return None

def get_lattice_parameters(castep_path):
    """
    Parse a CASTEP output file and extract, for each 'Unit Cell' block:
      - real_lattice: a 3×3 list of floats
      - a, b, c: lattice lengths (floats)
      - alpha, beta, gamma: cell angles in degrees (floats)
    
    Returns:
        List[dict] of the form:
        [
            {
                'real_lattice': [[r11, r12, r13],
                                 [r21, r22, r23],
                                 [r31, r32, r33]],
                'a': ...,
                'b': ...,
                'c': ...,
                'alpha': ...,
                'beta': ...,
                'gamma': ...,
            },
            ...
        ]
    """
    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    results = []
    with open(castep_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        # Look for the "Unit Cell" heading
        if 'Unit Cell' in lines[i]:
            # Advance to where the numeric lattice lines start
            j = i + 1
            # skip until we hit a line that looks like six floats
            while j < len(lines):
                parts = lines[j].split()
                if len(parts) >= 6 and all(is_float(p) for p in parts[:3]):
                    break
                j += 1
            if j >= len(lines) - 2:
                break

            # Read the 3×3 real lattice matrix
            real = []
            for k in range(j, j + 3):
                parts = lines[k].split()
                real.append([float(parts[0]), float(parts[1]), float(parts[2])])

            # Now find the lattice-parameters block (a, b, c and α, β, γ)
            # It always appears as three lines starting with "a =", "b =", "c ="
            # somewhere after our matrix, so scan forward a bit
            a = b = c = alpha = beta = gamma = None
            for k in range(j + 3, min(j + 20, len(lines))):
                if re.search(r'^\s*a\s*=', lines[k]):
                    # Expect three lines: a, b, c
                    for offset, (param, angle) in enumerate(
                        [('a','alpha'), ('b','beta'), ('c','gamma')]
                    ):
                        line = lines[k + offset]
                        # split on '=' then on whitespace
                        left, right = line.split('=', 1)
                        # right now like "   3.866591          alpha =   60.000000"
                        vals = right.replace('=',' ').split()
                        length = float(vals[0])
                        angle_val = float(vals[-1])
                        if param == 'a':
                            a, alpha = length, angle_val
                        elif param == 'b':
                            b, beta = length, angle_val
                        else:
                            c, gamma = length, angle_val
                    break

            # Save and advance
            results.append({
                'unit_cell': real,
                'a': a, 'b': b, 'c': c,
                'alpha': alpha, 'beta': beta, 'gamma': gamma,
            })
            i = j + 3
        else:
            i += 1

    return results

import re

def get_final_lattice_parameters(castep_path):
    """
    Parse a CASTEP output file and extract the *final* optimized cell,
    i.e. the first 'Unit Cell' block occurring after the
    'LBFGS: Final Configuration:' marker.
    
    Returns:
        dict with keys:
          - 'real_lattice': 3×3 list of floats
          - 'a', 'b', 'c': floats
          - 'alpha', 'beta', 'gamma': floats
    Raises:
        RuntimeError if no final configuration or no Unit Cell block is found.
    """
    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    with open(castep_path, 'r') as f:
        lines = f.readlines()

    # 1) Find the line index of the final‐configuration marker
    try:
        start = next(i for i, L in enumerate(lines)
                     if 'LBFGS: Final Configuration:' in L)
    except StopIteration:
        raise RuntimeError("No 'LBFGS: Final Configuration:' found in file")

    # 2) From there, find the next 'Unit Cell' header
    i = start + 1
    while i < len(lines) and 'Unit Cell' not in lines[i]:
        i += 1
    if i >= len(lines):
        raise RuntimeError("No Unit Cell block after final configuration")

    # 3) Skip forward to the numeric lattice rows
    j = i + 1
    while j < len(lines):
        parts = lines[j].split()
        if len(parts) >= 3 and all(is_float(p) for p in parts[:3]):
            break
        j += 1
    if j + 2 >= len(lines):
        raise RuntimeError("Incomplete lattice matrix after Unit Cell")

    # 4) Read the 3×3 real‐lattice matrix
    real = []
    for k in range(j, j + 3):
        p = lines[k].split()
        real.append([float(p[0]), float(p[1]), float(p[2])])

    # 5) Find a, b, c and α, β, γ in the following ~20 lines
    a = b = c = alpha = beta = gamma = None
    for k in range(j + 3, min(j + 20, len(lines))):
        if re.match(r'\s*a\s*=', lines[k]):
            # Expect three lines: a, b, c
            for offset, (param, angle) in enumerate(
                [('a','alpha'), ('b','beta'), ('c','gamma')]
            ):
                line = lines[k + offset]
                _, right = line.split('=', 1)
                vals = right.replace('=',' ').split()
                length = float(vals[0])
                angle_val = float(vals[-1])
                if param == 'a':
                    a, alpha = length, angle_val
                elif param == 'b':
                    b, beta = length, angle_val
                else:
                    c, gamma = length, angle_val
            break

    if None in (a, b, c, alpha, beta, gamma):
        raise RuntimeError("Failed to parse lattice parameters a/b/c/α/β/γ")

    unit_cell = real
    return unit_cell, a, b, c, alpha, beta, gamma


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



def get_final_fractional_positions(castep_path):
    """
    Extracts the final fractional positions from the LBFGS: Final Configuration block
    in a CASTEP .castep file, by looking for the specific border line
    'x----...----x' after the headers, then reading subsequent 'x ... x' lines.
    """
    lines = Path(castep_path).read_text().splitlines()
    frac_positions = []

    in_lbfgs = False
    start_parsing = False

    for line in lines:
        # 1) Enter LBFGS block
        if not in_lbfgs and "LBFGS: Final Configuration" in line:
            in_lbfgs = True
            continue

        if not in_lbfgs:
            continue

        # 2) Look for the dashed border after column headings:
        #    must start (after whitespace) with 'x-' and end with '-x'
        if not start_parsing:
            if re.match(r'^\s*x-+x\s*$', line):
                start_parsing = True
            continue

        # 3) Once parsing, stop on blank or non-x lines
        if not line.strip() or not line.lstrip().startswith('x'):
            break

        # 4) Strip off the 'x' borders and whitespace
        entry = line.strip().strip('x').strip()
        parts = entry.split()
        # Expect at least 5 fields: Element, Number, u, v, w
        if len(parts) < 5:
            continue

        symbol = parts[0]
        try:
            u, v, w = map(float, parts[2:5])
        except ValueError:
            continue

        frac_positions.append((symbol, u, v, w))

    return frac_positions

def fractional_coords_from_castep(castep_path):
    # 1. extract fractional positions and lattice
    fracs = extract_final_fractional_positions(castep_path)   # returns [(symbol,u,v,w),…]
    lat   = extract_lattice_parameters(castep_path)           # {'a':…, 'b':…, 'c':…, …}

    # 2. clean up symbols and build lists
    symbols = [s.split(':')[0] for s, u, v, w in fracs]  # drop any “:D” suffix
    scaled_positions = [(u, v, w) for s, u, v, w in fracs]

    # 3. assemble cell matrix (orthogonal example)
    a, b, c = lat['ax'], lat['ay'], lat['az']
    cell = [[a, 0, 0],
            [0, b, 0],
            [0, 0, c]]

    # 4. build the Atoms object
    atoms = Atoms(symbols=symbols,
                  scaled_positions=scaled_positions,
                  cell=cell,
                  pbc=True)
    
    return atoms 

# ============================================================================
#  Plotting and viewing functions
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

def view_structure(atoms,show_structure=True):
    view = nv.show_ase(atoms,scale=0.01, aspectRatio=1.)
    view.camera = 'orthographic'
    view.center()
    view.control.spin([0, 0, 1], math.pi/2)  # Rotate 90° about z-axis
    view.control.spin([0, 1, 0], math.pi/2)  # Rotate 90° about z-axis
    view.control.zoom(0.5)
    if show_structure:
        display(view)
        return
    else:
        return view
    
def plot_sequence(
    y,
    x=None,
    xlabel='Index',
    ylabel='Value',
    title='',
    figsize=(4,2),
    marker='o',
    **plot_kwargs
):
    """
    Plot a 1D sequence.
    
    Parameters
    ----------
    y : sequence of float
        The y-values to plot.
    x : sequence of float, optional
        The x-values. If None, uses range(len(y)).
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    title : str, optional
        Plot title.
    **plot_kwargs
        Additional keyword args passed to plt.plot (e.g. linestyle='--').
    """
    if x is None:
        x = list(range(len(y)))
    
    plt.figure(figsize=figsize)
    plt.plot(x, y, marker=marker, **plot_kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()




# ============================================================================
#  Macro like functions doing multiple things
# ============================================================================
def optimisation_summary_macro_1(castep_paths):
    for castep_path in castep_paths:
        # Header and error information
        print_filename(castep_path)
        #ct.extract_warnings(castep_path,verbose=True)

        # Energy convergence
        convergence = extract_LBFGS_energies(castep_path)
        final_enthalpy = extract_LBFGS_final_enthalpy(castep_path)
        print('Final enthalpy = {} eV.'.format(final_enthalpy))
        plot_energy_vs_iteration(convergence, title=castep_path.stem+' '+str(final_enthalpy),figsize=(5,2))
        
        # Unit cell parameters
        cell = extract_lattice_parameters(castep_path,a0=3.8668346, vac=15.0)
        cell_df = pd.DataFrame(cell.items(), columns=["Cell parameters", "Value"])
        display(cell_df) 

        # General parameters
        general_params =  extract_summary_parameters(castep_path)
        general_params_df = pd.DataFrame(general_params.items(), columns=["General parameter", "Value"])
        display(general_params_df) 

        # Show structure
        atoms = fractional_coords_from_castep(castep_path)
        view = view_structure(atoms,show_structure=False)
        display(view)
        img = view.render_image(frame=None, factor=4, antialias=True, trim=False, transparent=False)
        display(img)
        
# ============================================================================
#  Generate CASTEP cell files
# ============================================================================

def write_block_lattice_cart(lattice_cart,na=1,nb=1,nc=1):
    """
    Write a block lattice in Cartesian coordinates.
    """
    factors = np.array([[na, nb, nc]]).T  # shape (3,1)
    scaled_lattice_cart  = lattice_cart * factors

    lines = ["%BLOCK lattice_cart", "   ANG"]
    for row in scaled_lattice_cart:
        # format each number to 14 decimal places, aligned in 16-char fields
        fields = [f"{val:16.10f}" for val in row]
        # prefix with exactly 4 spaces before the first field
        lines.append("   " + "".join(fields))

    lines.append("%ENDBLOCK lattice_cart")

    block_text = "\n".join(lines)
    return(block_text)

def write_cell_constraints(constraints=None):
    """
    Write a block of cell constraints.

    %BLOCK CELL_CONSTRAINTS
        Ia Ib Ic
        Ialpha Ibeta Igamma
    %ENDBLOCK CELL_CONSTRAINTS

    - Entries 1–3 fix or couple the magnitudes a, b, c.
    - Entries 4–6 fix or couple the angles α, β, γ.
    - Zero means “fixed.”  Identical positive integers mean “tied together.”
    - A magnitude index (1–3) cannot appear also in the angle indices (4–6).
    """
    # 1) Default if none provided
    if constraints is None:
        constraints = [
            [0, 0, 0],
            [0, 0, 0],
        ]

    # 2) Normalize & validate
    constraints = np.asarray(constraints, dtype=int)
    if constraints.shape != (2, 3):
        raise ValueError(
            f"constraints must be shape (2,3), got {constraints.shape}"
        )

    # 3) Build the lines
    lines = ["%BLOCK CELL_CONSTRAINTS"]
    for row in constraints:
        # each integer in a 4-char field, space-separated
        fields = [f"{int(val):4d}" for val in row]
        lines.append("    " + " ".join(fields))
    lines.append("%ENDBLOCK CELL_CONSTRAINTS")

    return "\n".join(lines)


import numpy as np

def write_positions_frac(
    na=1,
    nb=1,
    nc=1,
    positions_frac=None
):
    """
    Generate a fractional-coordinate supercell block.

    Parameters
    ----------
    na, nb, nc : int
        Number of repetitions along x, y, z.
    positions_frac : array-like of shape (M,4), optional
        Each row is [atom_label, frac_x, frac_y, frac_z].
        If None, uses the default 4-atom Si (001) basis.

    Returns
    -------
    str
        A text block in the form:
        %BLOCK positions_frac
           Atom   x1      y1      z1
           ...
        %ENDBLOCK positions_frac
    """
    # -- set up the default 4-atom Si basis if needed
    if positions_frac is None:
        # list of tuples: (atom_label, x, y, z)
        positions_frac = [
            ("Si", 0.0,  0.0,  0.0),
            ("Si", 0.5,  0.0,  0.25),
            ("Si", 0.5,  0.5,  0.5),
            ("Si", 0.0,  0.5,  0.75),
        ]
    # -- otherwise we expect an array-like of shape (M,4)
    #    where element [i,0] is a string label, and [i,1:4] are fractional coords
    positions_frac = np.asarray(positions_frac, dtype=object)
    if positions_frac.ndim != 2 or positions_frac.shape[1] != 4:
        raise ValueError("positions_frac must be array-like of shape (M,4) "
                         "(atom_label, frac_x, frac_y, frac_z)")

    # split into labels and coords
    labels = positions_frac[:, 0].astype(str)
    coords = positions_frac[:, 1:].astype(float)

    super_cell = []
    super_labels = []

    # tile the basis over each (i,j,k) cell
    for i in range(na):
        for j in range(nb):
            for k in range(nc):
                # shift then normalize into supercell fractional coords
                # and carry along the atom label
                shifted = (coords + np.array([i, j, k])) / np.array([na, nb, nc])
                super_cell.append(shifted)
                super_labels.extend(labels)

    super_cell = np.vstack(super_cell)   # shape (M*na*nb*nc, 3)

    # build the text block
    lines = ["%BLOCK positions_frac"]
    for atom_label, (x, y, z) in zip(super_labels, super_cell):
        lines.append(
            f"   {atom_label:2s}   {x:16.10f}{y:16.10f}{z:16.10f}"
        )
    lines.append("%ENDBLOCK positions_frac")

    return "\n".join(lines)



def write_kpoints_mp_grid(kpoints_mp_grid):
    # only proceed if the user actually passed something
    if kpoints_mp_grid is not None:
        # ensure it’s an array (or list-like)
        arr = np.asarray(kpoints_mp_grid, dtype=int)

        # Option A: simple Python join
        parts = " ".join(str(x) for x in arr.tolist())
        return f"KPOINTS_MP_GRID : {parts}"
    else:
        return ""

def write_cell_file(
        title = None,
        path=".",
        filename="castep_input",
        na=1,
        nb=1,
        nc=1,
        lattice_cart=None,
        positions_frac=None,
        constraints=None,
        fix_all_ions=True,
        symmetry_generate=None,
        symmetry_tol = None,
        kpoints_mp_grid=None,
        display_file=True
    ):
    """
    Generate lattice, constraints and fractional positions for an
    nx ny nz supercell of `atom`, and write them all to a single file.
    
    Parameters
    ----------
    nx, ny, nz : int
        Number of repetitions along x, y, z.
    atom : str
        Element symbol (e.g. "Si").
    filename : str
        Name of the output file (e.g. "cell.in").
    path : str or Path, optional
        Directory to write into (default: current directory).
    a : float, optional
        Lattice constant to pass to write_block_lattice_cart.
    """
    # 
    if lattice_cart is None:
        lattice_cart = np.array([
            [2.7,     2.7,     0.0],
            [2.7,     0.0,     2.7],
            [0.0,     2.7,     2.7,]
        ])
    lattice_block = write_block_lattice_cart(lattice_cart, na=na, nb=nb, nc=nc)

    constraint_block = write_cell_constraints(constraints=constraints)
    positions_frac_block = write_positions_frac(na=na, nb=nb, nc=nc, positions_frac=positions_frac)
    kpoints_mp_grid_block = write_kpoints_mp_grid(kpoints_mp_grid)
    
    if symmetry_generate:
        symmetry_block = "SYMMETRY_GENERATE"
        if symmetry_tol:
            symmetry_block += "\nSYMMETRY_TOL : "+str(symmetry_tol)
    else: 
        symmetry_block = ""

    if fix_all_ions:
        fix_all_ions_block = "FIX_ALL_IONS : TRUE"
    else: 
        fix_all_ions_block = ""

    if title is not None:
        title_block = f"! {title}"
    else: 
        title_block = "! CASTEP cell file generated by SRSCALCUTILS"

    # 2) Concatenate with blank lines between
    full_text = "\n\n".join([
        title_block,
        lattice_block, 
        constraint_block, 
        positions_frac_block,
        symmetry_block,
        fix_all_ions_block,
        kpoints_mp_grid_block,
        "",
        ])
    
    # 3) Ensure output directory exists
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    filename = filename + ".cell"
    outfile = outdir / filename
    
    # 4) Write to disk
    with open(outfile, "w") as f:
        f.write(full_text)
    
    print(f"Wrote cell file to: {outfile}")

    if display_file:
        with open(outfile, "r") as f:
            print(f.read())

    return outfile

# ============================================================================
#  Generate CASTEP param files
# ============================================================================

def write_param_file(
        params,
        title=None,
        filename='crystal',
        path=".",
        display_file=False
    ):
    """
    Write a CASTEP .param file.

    Parameters
    ----------
    params : dict
        Mapping of keyword → value, e.g.
          {
            "TASK": "GeometryOptimization",
            "XC_FUNCTIONAL": "PBE",
            "CUT_OFF_ENERGY": 750,
            "SPIN_POLARISED": True,
            ...
          }
    title : str, optional
        If given, will be written as a commented title line:
        !TITLE: {title}
    filename : str
        Name of the output .param file.
    path : str or Path
        Directory in which to write.
    display_file : bool
        If True, print the contents after writing.
    """
    # 1) Prepare title line (commented)
    lines = []
    if title:
        lines.append(f"!TITLE: {title}")
        lines.append("")  # blank line after title

    # 2) Compute key column width for alignment
    #    uppercase the keys for consistency with CASTEP docs
    keys = [str(k).upper() for k in params]
    width = max(len(k) for k in keys)

    # 3) Format each param line
    for k in keys:
        v = params[k.lower()] if k.lower() in params else params[k]
        lines.append(f"{k:<{width}} : {v}")

    lines.append("\n")

    # 4) Join into full text
    full_text = "\n".join(lines)

    # 5) Ensure output directory exists
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    filename = filename + ".param"
    outfile = outdir / filename

    # 6) Write to disk
    with open(outfile, "w") as f:
        f.write(full_text)

    print(f"Wrote param file to: {outfile}")

    # 7) Optionally display
    if display_file:
        print(full_text)

    return outfile


# ============================================================================
#  Generate MYRIAD job submission scripts
# ============================================================================

def write_job_script(
    path,
    filename,
    wall_time='24:00:00',
    mem='10G',
    tmpfs='10G',
    n_procs=4,
    display_file=False
):
    """
    Write a job submission script for SGE with the given parameters.

    Args:
        path (str or Path): Directory where the .job file will be created.
        filename (str): Base name for the job file (no extension).
        wall_time (str): Wallclock time in format HH:MM:SS. Default '48:00:00'.
        mem (str): Memory per process (e.g., '10G'). Default '10G'.
        tmpfs (str): TMPDIR space per node (e.g., '20G'). Default '20G'.
        n_procs (int): Number of processors. Default 8.
        display_file (bool): If True, print the script contents after writing.

    Returns:
        Path: Path to the written .job file.
    """
    # Ensure output directory exists
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build file path
    job_file = out_dir / f"{filename}.job"

    # Script content
    content = f"""#!/bin/bash

# Request shell
#$ -S /bin/bash

# Request wallclock time (format hours:minutes:seconds).
#$ -l h_rt={wall_time}

# Request X gigabyte of RAM per process.
#$ -l mem={mem}

# Request TMPDIR space per node
#$ -l tmpfs={tmpfs}

# Set the working directory to be the directory the job is submitted from
#$ -cwd

# Set the name of the job.
#$ -N {filename}

# Merge .e and .o files (error and output)
#$ -j y

# Number of processors
#$ -pe mpi {n_procs}

# Setup the CASTEP calculation.
module load --redirect default-modules
module unload -f compilers mpi
module load mpi/intel/2019/update4/intel
module load compilers/intel/2019/update4
module load castep/19.1.1/intel-2019

# Run the CASTEP calculation

echo -n "Starting CASTEP calculation: "
date
gerun castep.mpi {filename}
echo -n "Finished: "
date
"""

    # Write to disk
    with open(job_file, 'w') as f:
        f.write(content)

    # Optionally display to console
    if display_file:
        print(content)

    return job_file
