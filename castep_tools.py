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
#region Module dependencies
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
from collections import defaultdict
from typing import List, Union, Tuple, Any


from IPython.display import display, Image as StaticImage
import time

# ============================================================================
#endregion Module dependencies
# ============================================================================


# ============================================================================
#region Basic file io and handling
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


def load_atoms_from_castep(path, filename):
    """
    Read both the 'positions_frac' and 'lattice_cart' blocks
    from a CASTEP-style text file.

    Returns
    -------
    positions_frac : np.ndarray, shape (N,4), dtype=object
        Rows of [element_label, x, y, z].
    lattice_cart : np.ndarray, shape (3,3), dtype=float
        The three Cartesian lattice vectors (in Å).
    """
    filepath = os.path.join(path, filename)
    positions = []
    lattice = []
    inside_pos = False
    inside_lat = False

    with open(filepath, 'r') as f:
        for line in f:
            s = line.strip()

            # detect start of one of our blocks
            if not (inside_pos or inside_lat):
                if s.startswith('%BLOCK'):
                    if 'positions_frac' in s:
                        inside_pos = True
                    elif 'lattice_cart' in s:
                        inside_lat = True
                continue

            # parse positions_frac block
            if inside_pos:
                if s.startswith('%ENDBLOCK') and 'positions_frac' in s:
                    inside_pos = False
                    continue
                if s:
                    parts = s.split()
                    elem = parts[0]
                    x, y, z = map(float, parts[1:4])
                    positions.append([elem, x, y, z])
                continue

            # parse lattice_cart block
            if inside_lat:
                # skip the "ANG" line
                if s.upper() == 'ANG':
                    continue
                if s.startswith('%ENDBLOCK') and 'lattice_cart' in s:
                    inside_lat = False
                    continue
                if s:
                    vec = list(map(float, s.split()))
                    lattice.append(vec)
                continue

    positions_frac = np.array(positions, dtype=object)
    lattice_cart  = np.array(lattice, dtype=float)
    return positions_frac, lattice_cart


def read_positions_frac_from_template(
        path=".",
        filename="filename", 
        lattice_cart_bulk=np.array([
            [3.8641976,     0.0,     0.0],
            [0.0,     7.7283952,     0.0],
            [0.0,     0.0,     5.4648012]
        ]),
        positions_frac_bulk = np.array([   
            ['Si',       0.0000000000,    0.7500000000,    0.7500000000],
            ['Si',       0.0000000000,    0.2500000000,    0.7500000000],
            ['Si',       0.5000000000,    0.7500000000,    0.5000000000],
            ['Si',       0.5000000000,    0.2500000000,    0.5000000000],
            ['Si',       0.5000000000,    0.5000000000,    0.2500000000],
            ['Si',       0.5000000000,    0.0000000000,    0.2500000000],
            ['Si',       0.0000000000,    0.5000000000,    0.0000000000],
            ['Si',       0.0000000000,    0.0000000000,    0.0000000000]
            ], dtype=object),
        surface_unit_cell_dims = np.array([1,1,2]), 
        sort_order=['z', 'y', 'x', 'atom']
    ):
    """
    Read positions in fractional coordinates from a template file.
    """
    # Load the template file and make sure it is sorted
    positions_frac_surf, lattice_cart_surf = load_atoms_from_castep(path, filename)
    positions_frac_surf = sort_positions_frac(positions_frac_surf, order=sort_order)

    # Convert template to cartesian coordinates
    positions_cart_surf = frac_to_cart(lattice_cart_surf,positions_frac_surf)

    # Calculate the number of atoms in the surface unit cell to take from the template
    number_of_atoms_surf = positions_frac_bulk.shape[0] * surface_unit_cell_dims[2]

    # Select the correct number of atoms from the template
    positions_cart_surf = positions_cart_surf[:number_of_atoms_surf,:]

    # Adjust the lattice vectors for the surface unit cell (multiply the z-dimension)
    lattice_cart_surf = lattice_cart_bulk.copy()
    lattice_cart_surf[-1, :] *= surface_unit_cell_dims[2]  # Adjust the z-dimension

    # Remove the z-offset from the fractional coordinates
    positions_cart_surf = remove_z_offset(positions_cart_surf)   

    # Convert the selected positions back to fractional coordinates
    positions_frac_surf = cart_to_frac(lattice_cart_surf, positions_cart_surf)

    return positions_frac_surf, lattice_cart_surf


def delete_all_files_in_cwd(force: bool = False):
    cwd = Path('.').resolve()
    files = [f for f in cwd.iterdir() if f.is_file()]
    if not files:
        print(f"No files found in {cwd}. Nothing to delete.")
        return

    # If not forcing, show and prompt
    if not force:
        print(f"Found {len(files)} file(s) in {cwd}:")
        for f in files:
            print(f"  • {f.name}")

        confirm = input("Delete ALL these files? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Aborted. No files were deleted.")
            return

    # Proceed with deletion (forced or user-confirmed)
    deleted = 0
    for entry in files:
        try:
            entry.unlink()
            deleted += 1
        except Exception as e:
            print(f"Error deleting {entry.name}: {e}")

    print(f"Done. {deleted} file(s) deleted.")


# ============================================================================
#endregion Basic file io and handling
# ============================================================================


# ============================================================================
#region Get information from .castep files
# ============================================================================


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


def get_lattice_parameters(castep_path, a0=3.8668346, vac=15.0):
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
    fracs = get_final_fractional_positions(castep_path)   # returns [(symbol,u,v,w),…]
    lat   = get_lattice_parameters(castep_path)           # {'a':…, 'b':…, 'c':…, …}

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
#endregion Get information from .castep files
# ============================================================================


# ============================================================================
#region Plotting and visualization functions
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
#endregion Plotting and visualization functions
# ============================================================================

        
# ============================================================================
#region Writing CASTEP input files
# ============================================================================

def write_block_lattice_cart(lattice_cart):
    """
    Write a block lattice in Cartesian coordinates.
    """
    lines = ["%BLOCK lattice_cart", "   ANG"]
    for row in lattice_cart:
        # format each number to 14 decimal places, aligned in 16-char fields
        fields = [f"{val:16.10f}" for val in row]
        # prefix with exactly 4 spaces before the first field
        lines.append("   " + "".join(fields))

    lines.append("%ENDBLOCK lattice_cart")

    block_text = "\n".join(lines)
    return(block_text)

def write_cell_constraints(cell_constraints=None):
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
    if cell_constraints is None:
        cell_constraints = [
            [0, 0, 0],
            [0, 0, 0],
        ]

    # 2) Normalize & validate
    cell_constraints = np.asarray(cell_constraints, dtype=int)
    if cell_constraints.shape != (2, 3):
        raise ValueError(
            f"constraints must be shape (2,3), got {constraints.shape}"
        )

    # 3) Build the lines
    lines = ["%BLOCK CELL_CONSTRAINTS"]
    for row in cell_constraints:
        # each integer in a 4-char field, space-separated
        fields = [f"{int(val):4d}" for val in row]
        lines.append("    " + " ".join(fields))
    lines.append("%ENDBLOCK CELL_CONSTRAINTS")

    block_text = "\n".join(lines)
    return block_text


def write_positions_frac(
    positions_frac: np.ndarray
) -> str:
    """
    Generate a fractional-coordinate supercell block.

    Parameters
    ----------
    positions_frac : ndarray, shape (M,4), dtype object, optional
        Each row is [atom_label, frac_x, frac_y, frac_z].
        If None, uses the default 4-atom Si (001) basis.

    Returns
    -------
    str
        A text block formatted as:
        %BLOCK positions_frac
           Atom   x1      y1      z1
           ...
        %ENDBLOCK positions_frac
    """
    # Default Si (001) basis
    if positions_frac is None:
        positions_frac = np.array([
            ['Si', 0.0,  0.0,  0.0],
            ['Si', 0.5,  0.0,  0.25],
            ['Si', 0.5,  0.5,  0.5],
            ['Si', 0.0,  0.5,  0.75],
        ], dtype=object)

    # Build text block
    lines = ["%BLOCK positions_frac"]
    for atom_label, x, y, z in positions_frac:
        lines.append(f"   {atom_label:2s}   {float(x):16.10f}{float(y):16.10f}{float(z):16.10f}")
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
        lattice_cart=None,
        positions_frac=None,
        cell_constraints=None,
        ionic_constraints=None,
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
    lattice_block = write_block_lattice_cart(lattice_cart)
    cell_constraint_block = write_cell_constraints(cell_constraints=cell_constraints)
    positions_frac_block = write_positions_frac(positions_frac=positions_frac)
    ionic_constraints_block = write_ionic_constraints(ionic_constraints=ionic_constraints)
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

    blocks = [
        title_block,
        lattice_block, 
        cell_constraint_block, 
        positions_frac_block,
        ionic_constraints_block,
        symmetry_block,
        fix_all_ions_block,
        kpoints_mp_grid_block]
    
    # filter out empty blocks
    blocks = [blk for blk in blocks if blk]
    blocks.append("")
    
    # 2) Concatenate with blank lines between
    full_text = "\n\n".join(blocks)
    
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


def write_ionic_constraints(ionic_constraints):
    """
    Generate CASTEP ionic constraint entries for a list of selected atoms,
    wrapped in BLOCK/ENDBLOCK.

    Parameters
    ----------
    atom_array : array-like, shape (N,6)
        Output from select_atoms_by_region:
        [input_index, is_selected, atom_symbol, frac_x, frac_y, frac_z]

    Returns
    -------
    lines : list of str
        Text lines beginning with '%BLOCK ionic_constraints',
        followed by three constraint lines per atom, ending with
        '%ENDBLOCK ionic_constraints'.
    """
    if ionic_constraints is None:
        return ""
    
    arr = np.array(ionic_constraints, dtype=object)

    # Extract columns
    is_selected = arr[:, 1].astype(bool)
    symbols = arr[:, 2]

    # Build running count per symbol
    counts = defaultdict(int)
    per_symbol_count = []
    for sym in symbols:
        counts[sym] += 1
        per_symbol_count.append(counts[sym])

    constraints = []
    out_idx = 1  # global row counter

    # Build raw constraint rows
    for sel, sym, sym_count in zip(is_selected, symbols, per_symbol_count):
        if not sel:
            continue
        for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            constraints.append((out_idx, sym, sym_count, dx, dy, dz))
            out_idx += 1

    # Format lines with BLOCK wrapper
    lines = ['%BLOCK ionic_constraints']
    for idx, sym, aid, dx, dy, dz in constraints:
        lines.append(
            f"{idx:5d} {sym:<2}        {aid:>2d}    {dx:10.8f}    {dy:10.8f}    {dz:10.8f}"
        )
    lines.append('%ENDBLOCK ionic_constraints')

    block_text = "\n".join(lines)
    return block_text


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


def write_xyz(positions_cart, path='.', filename='castep_input', comment=None):
    """
    Write a .xyz file from an array of symbols and Cartesian coordinates.

    Parameters
    ----------
    positions_cart : array-like, shape (N,4)
        Array of N atomic positions: each row is [element_symbol, x, y, z].
    path : str, optional
        Directory where the .xyz file will be written (default: current directory).
    filename : str, optional
        Base name for the output file (default: "castep_input").
    comment : str, optional
        Comment line in the .xyz file (default: blank).

    Returns
    -------
    xyz_str : str
        The contents of the generated .xyz file.
    """
    positions_cart = np.asarray(positions_cart, dtype=object)
    n = len(positions_cart)

    # Prepare output path
    out_fname = filename if filename.lower().endswith('.xyz') else filename + '.xyz'
    out_path = os.path.join(path, out_fname)

    # Header
    lines = [str(n), comment or '']

    # Atom lines
    for row in positions_cart:
        symbol = row[0]
        x, y, z = map(float, row[1:])
        lines.append(f"{symbol} {x:.6f} {y:.6f} {z:.6f}")

    xyz_str = "\n".join(lines) + "\n"

    # Write to disk
    with open(out_path, 'w') as f:
        f.write(xyz_str)
    
    print(f"Wrote XYZ file to: {out_path}")

    return xyz_str


# ============================================================================
#endregion Writing CASTEP input files
# ============================================================================


# ============================================================================
#region Manipulate arrays, lists, coordinates, and cells. 
# ============================================================================

def show_only_true(data):
    """
    Return only those rows from `data` (a list of rows) 
    where at least one element is the boolean True.
    """
    return [row for row in data if any(cell is True for cell in row)]


def labelled_positions_frac_to_positions_frac(labelled):
    """
    Given an array-like `labelled` where each row has two extra columns
    on the left (e.g. an index and a boolean flag), return a new NumPy
    array containing only the remaining columns (label + fractions).
    
    Example input rows:
      [1, True, 'Si', 0.0,  0.0,  0.0]
      [2, True, 'Si', 0.5,  0.0,  0.25]
      ...
    
    Output rows:
      ['Si', 0.0,  0.0,  0.0]
      ['Si', 0.5,  0.0,  0.25]
      ...
    """
    arr = np.asarray(labelled, dtype=object)
    # slice off the first two columns
    return arr[:, 2:]


def replace_element(data, old, new):
    """
    Replace all occurrences of `old` with `new` in a nested list or numpy array.
    Returns a new data structure with replacements, leaving the original intact.
    """
    # Handle numpy arrays
    if isinstance(data, np.ndarray):
        arr = data.copy()
        arr[arr == old] = new
        return arr
    
    # Handle plain Python nested lists
    if isinstance(data, list):
        return [
            [new if elem == old else elem for elem in row]
            for row in data
        ]
    
    raise TypeError(f"Unsupported data type: {type(data)}")


def create_supercell_from_fractional_coords(
    positions_frac: np.ndarray,
    lattice_cart: np.ndarray,
    n: np.ndarray
) -> np.ndarray:
    """
    Generate a supercell from fractional atomic positions.

    Parameters
    ----------
    positions_frac : ndarray, shape (M,4), dtype object
        Each row is [atom_label, frac_x, frac_y, frac_z].
    na, nb, nc : int
        Number of repetitions along x, y, z.

    Returns
    -------
    supercell_frac : ndarray, shape (M*na*nb*nc, 4), dtype object
        Each row is [atom_label, frac_x, frac_y, frac_z] in the supercell.
    """
    # Validate input
    positions_frac = np.asarray(positions_frac, dtype=object)
    if positions_frac.ndim != 2 or positions_frac.shape[1] != 4:
        raise ValueError(
            "positions_frac must be array-like of shape (M,4) "
            "(atom_label, frac_x, frac_y, frac_z)"
        )

    labels = positions_frac[:, 0]
    coords = positions_frac[:, 1:].astype(float)

    supercell_list = []

    na = n[0]
    nb = n[1]
    nc = n[2]

    for i in range(na):
        for j in range(nb):
            for k in range(nc):
                # shift then normalize
                shifted = (coords + np.array([i, j, k])) / np.array([na, nb, nc])
                # stack labels and shifted coords
                block = np.empty((shifted.shape[0], 4), dtype=object)
                block[:, 0] = labels
                block[:, 1:] = shifted
                supercell_list.append(block)

    supercell_frac = np.vstack(supercell_list)

    supercell_frac = sort_positions_frac(supercell_frac, order = ['z', 'y', 'x', 'atom'], descending=True)

    lattice_cart = (lattice_cart.T * n).T

    return supercell_frac, lattice_cart


def bond_vector_from_spherical(theta, phi, bondlength, tol=1e-8):
    """
    Convert spherical coordinates (physicist’s convention) to a 3D Cartesian bond vector,
    and zero out any component smaller than `tol`.

    Parameters
    ----------
    theta : float
        Polar angle in radians, from +z toward the xy-plane.  Range: 0 ≤ θ ≤ π.
    phi : float
        Azimuthal angle in radians, from +x toward +y.  Range: 0 ≤ φ < 2π.
    bondlength : float
        Radial distance (bond length), r ≥ 0.
    tol : float, optional
        Any |component| < tol will be set to exactly 0.0  (default = 1e-8).

    Returns
    -------
    bond : ndarray of shape (3,)
        Cartesian vector [x, y, z], where
            x = r * sin(theta) * cos(phi)
            y = r * sin(theta) * sin(phi)
            z = r * cos(theta)
        and any tiny numerical remnant smaller than `tol` is snapped to zero.
    """
    r = float(bondlength)
    sin_t = np.sin(theta)
    x = r * sin_t * np.cos(phi)
    y = r * sin_t * np.sin(phi)
    z = r * np.cos(theta)

    # snap near-zero values to exactly 0.0
    x = 0.0 if abs(x) < tol else x
    y = 0.0 if abs(y) < tol else y
    z = 0.0 if abs(z) < tol else z

    return np.array([x, y, z], dtype=float)


def add_atoms_to_positions_frac(
    labeled_positions_frac,
    lattice_cart,
    bond,
    atom="H",
    extend_unit_cell=(0, 0, 1)
):
    """
    Append offset atoms to a fractional-coordinate list, handling either a single bond vector
    or a list of bond vectors, with optional cell extension and periodic wrap.

    Parameters
    ----------
    labeled_positions_frac : sequence of [idx, flag, sym, fx, fy, fz]
        Original atoms with boolean flag for adding offsets.
    lattice_cart : (3,3) array-like
        Cartesian cell vectors (rows).
    bond : array-like, shape (3,) or (M,3)
        If shape (3,), a single offset vector. If shape (M,3), a list of M offset vectors.
    atom : str, optional
        Symbol for each new atom (default "H").
    extend_unit_cell : length-3 sequence of 0/1, optional
        Which axes to allow cell expansion on. E.g. (0,0,1) means only z may extend,
        (1,1,1) means x,y,z all may extend. Default (0,0,1).

    Returns
    -------
    positions_frac : ndarray, shape (N + M*K, 4), dtype=object
        Columns: [symbol, fx, fy, fz], all Python floats. K = # flagged atoms,
        M = # bond vectors (or 1).
    new_lattice_cart : ndarray (3,3)
        Cell vectors expanded only along permitted axes.
    """
    # 1) prepare bond vectors & fractional offsets
    bonds = np.atleast_2d(bond).astype(float)     # shape (M,3)
    inv_lat = np.linalg.inv(lattice_cart)
    bond_fracs = bonds @ inv_lat                 # shape (M,3)

    # 2) collect originals and all prospective new positions
    orig = []
    new_fracs = []
    for idx, flag, sym, fx, fy, fz in labeled_positions_frac:
        f = np.array([fx, fy, fz], float)
        orig.append((sym, f))
        if flag:
            for bf in bond_fracs:
                new_fracs.append(f + bf)
    new_fracs = np.vstack(new_fracs) if new_fracs else np.zeros((0,3))

    # 3) compute minimal fractional shift Δ to bring new_fracs into [0,1]
    low  = np.maximum(0.0, -new_fracs.min(axis=0))
    high = np.maximum(0.0,  new_fracs.max(axis=0) - 1.0)
    shift_frac = low - high

    # 4) mask shifts by allowed axes
    extend_axes = np.array(extend_unit_cell, dtype=int)
    shift_frac = shift_frac * extend_axes

    # 5) build output positions
    stretch = 1.0 + np.abs(shift_frac)
    out = []

    # 5a) Originals: shift & normalize (they stay inside [0,1])
    for sym, f in orig:
        f_final = (f + shift_frac) / stretch
        out.append([
            sym,
            float(f_final[0]),
            float(f_final[1]),
            float(f_final[2])
        ])

    # 5b) New atoms: shift, then clamp or wrap per axis, then normalize
    for f in new_fracs:
        f2 = f + shift_frac
        # per-axis clamp or wrap
        for i in range(3):
            if extend_axes[i]:
                # clamp into [0,1]
                f2[i] = min(max(f2[i], 0.0), 1.0)
            else:
                # periodic wrap
                f2[i] = f2[i] % 1.0
        f_final = f2 / stretch
        out.insert(0,[
            atom,
            float(f_final[0]),
            float(f_final[1]),
            float(f_final[2])
        ])

    positions_frac = np.array(out, dtype=object)

    # 6) expand the allowed cell vectors by |shift_frac|
    new_lattice_cart = lattice_cart.copy()
    for i in range(3):
        if extend_axes[i]:
            vec = lattice_cart[i]
            new_lattice_cart[i] = vec * (1.0 + abs(shift_frac[i]))

    return positions_frac, new_lattice_cart


# def dimerise_displacement(
#     labeled_positions_frac,
#     lattice_cart,
#     dimer_direction,
#     displacement_direction,
#     displacement=0.5,
#     start_phase='+',
#     alternate=False,
#     wrap_axes=(1,1,1)
# ):
#     """
#     Displace selected atoms in alternating directions (dimerisation):
#     pairing along `dimer_direction`, displacing along `displacement_direction`.

#     Parameters
#     ----------
#     labeled_positions_frac : sequence of [idx, flag, sym, fx, fy, fz]
#         Input atoms; only those with flag=True are processed.
#     lattice_cart : (3,3) array-like
#         Cartesian cell vectors (rows), used to convert Angstrom to fractional.
#     dimer_direction : {'x','y','z'}
#         Axis along which atoms are paired (chains formed by identical values
#         of the other two fractional coords).
#     displacement_direction : {'x','y','z'}
#         Axis along which to apply alternating displacement.
#     displacement : float, optional
#         Cartesian displacement magnitude in Angstrom (default=0.5).
#     start_phase : {'+','-'}, optional
#         Sign of the first displacement in the first chain (default='+').
#     alternate : bool, optional
#         If True, flip the start_phase for each new chain in order (default=False).
#     wrap_axes : length-3 sequence of 0/1, optional
#         Axes where periodic wrapping should apply (default=(1,1,1)).

#     Returns
#     -------
#     positions_frac : ndarray of shape (N, 4), dtype=object
#         Array of [symbol, fx, fy, fz] with updated fractional positions.

#     Raises
#     ------
#     ValueError
#         If any chain has an odd number of atoms or invalid directions.
#     """
#     # Map directions to indices
#     axes = {'x':0, 'y':1, 'z':2}
#     if dimer_direction not in axes or displacement_direction not in axes:
#         raise ValueError("`dimer_direction` and `displacement_direction` must be one of 'x','y','z'.")
#     dimer_ax = axes[dimer_direction]
#     disp_ax  = axes[displacement_direction]

#     # Convert displacement (Å) to fractional along disp_ax
#     disp_vec = np.zeros(3, dtype=float)
#     disp_vec[disp_ax] = displacement
#     inv_lat = np.linalg.inv(lattice_cart)
#     disp_frac = disp_vec @ inv_lat

#     # Prepare atom data
#     atoms = []
#     for idx, flag, sym, fx, fy, fz in labeled_positions_frac:
#         atoms.append({'flag': flag,
#                       'sym': sym,
#                       'coord': np.array([fx,fy,fz], float)})

#     # Group flagged atoms by the other two coords
#     groups = {}
#     for atom in atoms:
#         if not atom['flag']:
#             continue
#         key = tuple(np.delete(atom['coord'], dimer_ax))
#         groups.setdefault(key, []).append(atom)

#     # Check even sizes and apply alternating displacements
#     # Sort groups by key for consistent ordering
#     sorted_items = sorted(groups.items(), key=lambda kv: kv[0])
#     for idx, (key, grp) in enumerate(sorted_items):
#         if len(grp) % 2 != 0:
#             raise ValueError(f"Chain at {key} has odd number of atoms ({len(grp)})")
#         # Determine phase for this chain
#         phase = start_phase
#         if alternate and (idx % 2 == 1):
#             phase = '+' if start_phase == '-' else '-'
#         sign = 1 if phase == '+' else -1
#         # Sort along dimer axis and apply
#         grp.sort(key=lambda a: a['coord'][dimer_ax])
#         for atom in grp:
#             atom['coord'][disp_ax] += sign * disp_frac[disp_ax]
#             sign *= -1

#     # Build output with wrapping/clamping
#     wrap = np.array(wrap_axes, int)
#     out = []
#     for atom in atoms:
#         c = atom['coord']
#         for i in range(3):
#             if wrap[i]:
#                 c[i] %= 1.0
#             else:
#                 c[i] = min(max(c[i], 0.0), 1.0)
#         out.append([atom['sym'], float(c[0]), float(c[1]), float(c[2])])

#     return np.array(out, dtype=object)


def create_surface_supercell(
    lattice_cart_bulk: np.ndarray,
    positions_frac_bulk: np.ndarray,
    positions_frac_surf: np.ndarray,
    n: tuple[int,int,int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an (na x nb x nc) supercell whose bottom 1/nc is the bulk motif
    and whose top (z_multiple/nc) is the surface motif.

    Returns
    -------
    supercell_frac : (M*na*nb, 4) object-array
      Fractional [label, x,y,z] in the supercell.
    lattice_cart_super : (3,3) float-array
    The Cartesian lattice = bulk_lattice * diag(na,nb,nc).
    """
    # Get the unit cell repeat numbers
    na, nb, nc = n

    # Check user input - can't make a cell less that 2x in z
    if nc < 2:
        raise ValueError("n[2] (z-multiple) must be at least 2 for a surface supercell.")
    
    # Get the number of atoms in the bulk and surface cells
    atoms_bulk = positions_frac_bulk.shape[0]
    atoms_surf = positions_frac_surf.shape[0]

    # Calculate the integer ratio of the bulk cell in the surface cell by atomno
    if atoms_surf % atoms_bulk == 0:
        surfbulkratio = atoms_surf // atoms_bulk
    else:
        raise ValueError("The number of atoms in the surface motif must be a multiple of the number of atoms in the bulk motif.")

    # Calculate the repeat numbers for surface and bulk unit cells depending on size of surface cell and repeat numbers chosen
    if nc > surfbulkratio:
        n_bulk = nc - surfbulkratio
    elif nc <= surfbulkratio:
        n_bulk = 1
        atoms_surf = atoms_surf - atoms_bulk * (nc - 1)
        # make the surface fractional positions list smaller
        blocks = []
        for idx, (label, x, y, z) in enumerate(positions_frac_surf):
            if idx >= atoms_surf:
                break
            blocks.append((label, x, y, z))
        positions_frac_surf = np.array(blocks)
        positions_frac_surf = remove_z_offset(positions_frac_surf, decimals=8)
    
    # Make sure the surface unit cell atoms are correctly ordered
    positions_frac_surf = sort_positions_frac(positions_frac_surf, order=['z', 'y', 'x', 'atom'])

    # Rescale the bulk coordinates. 
    positions_frac_bulk = [
        (label, x / na, y /nb, z /nc)
        for (label, x, y, z) in positions_frac_bulk
    ]   

    # Rescale the surface coordinates. 
    scale = surfbulkratio / nc 
    positions_frac_surf = [
        (label, x, y, z * scale)
        for (label, x, y, z) in positions_frac_surf
    ]   

    blocks = []
    # bulk atoms
    for i in range(n_bulk):
        for label, x, y, z in positions_frac_bulk:
            z_new = i / nc + z
            blocks.append((label, x, y, z_new))
    #surface atoms
    for idx, (label, x, y, z) in enumerate(positions_frac_surf):
        if idx >= atoms_surf:
            break
        z_new = (n_bulk / nc) + z
        blocks.append((label, x / na, y / nb, z_new))
    
    positions_frac_super = np.array(blocks, dtype=object)
    
    # Now calculate repeats in x and y
    blocks = []
    for i in range(na):
            for j in range(nb):
                for label, x, y, z in positions_frac_super:
                    blocks.append((label, i / na + x , j / nb +y , z))

    positions_frac_super = np.array(blocks, dtype=object)
    positions_frac_super = sort_positions_frac(positions_frac_super, order=['z', 'y', 'x', 'atom'])

    # Calculate the final unit cell
    lattice_cart_super = lattice_cart_bulk.copy()
    lattice_cart_super[0] = lattice_cart_super[0] * na
    lattice_cart_super[1] = lattice_cart_super[1] * nb
    lattice_cart_super[2] = lattice_cart_super[2] * nc

    return lattice_cart_super, positions_frac_super


def create_vacuum_spacing(
    positions_frac: np.ndarray,
    lattice_cart: np.ndarray,
    vac: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Insert a vacuum layer above a unit cell.

    Parameters
    ----------
    positions_frac : ndarray, shape (M,4), dtype object
        Each row is [atom_label, frac_x, frac_y, frac_z] of the original cell.
    lattice_cart : ndarray, shape (3,3), float
        The Cartesian lattice vectors as rows: [a⃗; b⃗; c⃗].
    vac : float
        Vacuum thickness to add along the c‐axis (in Å).

    Returns
    -------
    new_positions_frac : ndarray, shape (M,4), dtype object
        [atom_label, frac_x, frac_y, frac_z_new] with z‐fractions scaled by orig_c/(orig_c+vac).
    new_lattice_cart : ndarray, shape (3,3), float
        Same a⃗ and b⃗, but c⃗ lengthened by vac.
    """
    # -- validate inputs
    pos = np.asarray(positions_frac, dtype=object)
    if pos.ndim!=2 or pos.shape[1]!=4:
        raise ValueError("positions_frac must be shape (M,4): [label, fx, fy, fz]")
    L = np.asarray(lattice_cart, dtype=float)
    if L.shape!=(3,3):
        raise ValueError("lattice_cart must be shape (3,3)")

    labels = pos[:,0]
    coords = pos[:,1:].astype(float)

    # -- original c-vector and its length
    c_vec = L[2]
    orig_c = np.linalg.norm(c_vec)
    if orig_c == 0:
        raise ValueError("c-axis length is zero!")
    c_dir = c_vec / orig_c

    # -- build new lattice: only extend along c_dir
    new_c = (orig_c + vac) * c_dir
    new_L = L.copy()
    new_L[2] = new_c

    # -- scale down all fractional z so atoms sit within [0, orig_c/(orig_c+vac)]
    scale = orig_c / (orig_c + vac)
    new_coords = coords.copy()
    new_coords[:, 2] *= scale

    # -- pack back into object array
    M = coords.shape[0]
    new_positions_frac = np.empty((M, 4), dtype=object)
    new_positions_frac[:, 0] = labels
    new_positions_frac[:, 1:] = new_coords

    return new_positions_frac, new_L


def sort_positions_frac(arr: np.ndarray,
                        order: list[str] = ['z', 'y', 'x', 'atom'],
                        descending: bool = True) -> np.ndarray:
    """
    Sorts an array of shape (N, 4) with dtype=object based on specified fields.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (N, 4), with columns ['atom', 'x', 'y', 'z'] and dtype=object.
    order : list[str], optional
        List of field names to sort by, in priority order (highest first).
        Supported names: 'atom', 'x', 'y', 'z'. Defaults to ['z', 'y', 'x', 'atom'].
    descending : bool, optional
        If True, sort from highest to lowest along the specified order;
        if False, sort from lowest to highest. Default is False.

    Returns
    -------
    np.ndarray
        New sorted array of the same shape and dtype.
    """
    # Supported fields for sorting
    FIELD_INDICES = {
        'atom': 0,
        'x': 1,
        'y': 2,
        'z': 3,
    }
    # Validate order list
    for field in order:
        if field not in FIELD_INDICES:
            raise ValueError(f"Unsupported sort field: {field}. "
                             f"Choose from {list(FIELD_INDICES.keys())}.")

    # Convert to list of rows for sorting
    rows = arr.tolist()

    # Create a tuple key based on requested fields
    def sort_key(row: list) -> tuple:
        return tuple(row[FIELD_INDICES[f]] for f in order)

    # Sort and return; reverse if descending
    rows_sorted = sorted(rows, key=sort_key, reverse=descending)
    return np.array(rows_sorted, dtype=object)


def remove_z_offset(positions_frac, decimals=7):
    """
    Shift all z‐coordinates so that the minimum becomes zero,
    and round all fractional coords to at most `decimals` places.

    Parameters
    ----------
    positions_frac : array‐like, shape (N,4), dtype object
        Rows of [element_symbol, x_frac, y_frac, z_frac].
    decimals : int
        Maximum number of decimal places for x,y,z. Default is 7.

    Returns
    -------
    new_positions : ndarray, shape (N,4), dtype object
        Same as input but with every z_frac replaced by (z_frac – z_min),
        and x_frac, y_frac, z_frac rounded to `decimals`.
    """
    arr = np.asarray(positions_frac, dtype=object)

    # extract columns
    elems = arr[:, 0]
    coords = arr[:, 1:].astype(float)  # shape (N,3)

    # compute and subtract z‐offset
    z_min = coords[:, 2].min()
    coords[:, 2] = coords[:, 2] - z_min

    # round all three columns
    coords = np.round(coords, decimals)

    # reassemble
    out = np.empty_like(arr)
    out[:, 0] = elems
    out[:, 1:] = coords

    return out



def merge_posfrac_or_labelled_posfrac(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """
    Merge two 2D arrays row‐wise, preserving the first occurrence of any duplicate row.

    Parameters
    ----------
    a1, a2 : np.ndarray
        Input arrays of shape (n1, m) and (n2, m). Rows are compared element‐by‐element
        (so all row elements must be hashable: e.g. numbers, strings, bools).

    Returns
    -------
    np.ndarray
        An array containing all unique rows from a1 then a2 (in order of first appearance),
        shape (m, #cols), dtype same as the inputs.
    """
    # sanity check
    if a1.ndim != 2 or a2.ndim != 2 or a1.shape[1] != a2.shape[1]:
        raise ValueError(f"Both inputs must be 2D arrays with the same number of columns, "
                         f"got {a1.shape} and {a2.shape}")

    # stack them
    combined = np.vstack((a1, a2))

    seen = set()
    unique_rows = []
    for row in map(tuple, combined):
        if row not in seen:
            seen.add(row)
            unique_rows.append(row)

    return np.array(unique_rows, dtype=a1.dtype)


def frac_to_cart(lattice_cart, positions_frac, tol=1e-6):
    """
    Convert fractional coordinates to Cartesian and clean up small residues.

    Parameters
    ----------
    lattice_cart : array-like, shape (3,3)
        Lattice vectors in Cartesian coordinates.
    positions_frac : array-like, shape (N,4)
        Rows of [element_symbol, f_x, f_y, f_z].
    tol : float, optional
        Any Cartesian coordinate with abs(value) < tol is zeroed. Default 1e-6.

    Returns
    -------
    positions_cart : ndarray, shape (N,4), dtype object
        Rows of [element_symbol, x, y, z], where x,y,z are floats rounded to 6 dp.
    """
    lattice_cart = np.asarray(lattice_cart, dtype=float)
    positions_frac = np.asarray(positions_frac, dtype=object)

    n = len(positions_frac)
    positions_cart = np.empty((n, 4), dtype=object)

    for i, row in enumerate(positions_frac):
        symbol = row[0]
        frac = np.array(row[1:], dtype=float)
        cart = frac.dot(lattice_cart)

        # zero out tiny residuals below tol
        cart[np.abs(cart) < tol] = 0.0

        # round to 7 decimal places
        cart = np.round(cart, 7)

        positions_cart[i, 0] = symbol
        positions_cart[i, 1:] = cart

    return positions_cart


def cart_to_frac(lattice_cart, positions_cart, tol=1e-6):
    """
    Convert Cartesian coordinates back to fractional coordinates,
    zeroing out any small residues and rounding to 6 decimal places.

    Parameters
    ----------
    lattice_cart : array-like, shape (3,3)
        Lattice vectors in Cartesian coordinates (rows are a, b, c).
    positions_cart : array-like, shape (N,4)
        Rows of [element_symbol, x, y, z] in Cartesian coords.
    tol : float, optional
        Any fractional coordinate with abs(value) < tol is zeroed. Default 1e-6.

    Returns
    -------
    positions_frac : ndarray, shape (N,4), dtype object
        Rows of [element_symbol, f_x, f_y, f_z], with floats rounded to 6 dp.
    """
    # ensure float array and invert
    lat = np.asarray(lattice_cart, dtype=float)
    inv_lat = np.linalg.inv(lat)

    pcs = np.asarray(positions_cart, dtype=object)
    n = len(pcs)
    positions_frac = np.empty((n, 4), dtype=object)

    for i, row in enumerate(pcs):
        symbol = row[0]
        cart = np.array(row[1:], dtype=float)

        # fractional = cart · inv(lat)
        frac = cart.dot(inv_lat)

        # zero-out tiny values
        frac[np.abs(frac) < tol] = 0.0

        # round to 6 decimal places
        frac = np.round(frac, 6)

        positions_frac[i, 0] = symbol
        positions_frac[i, 1:] = frac

    return positions_frac


def select_atoms_by_region(positions_frac, lattice_cart, condition,
                           include=None, exclude=None):
    """
    Select atoms by a region defined in Cartesian space, with options to include or exclude by index.

    Parameters
    ----------
    positions_frac : array-like, shape (N, 4)
        Rows of [element_symbol, frac_x, frac_y, frac_z].
    lattice_cart : array-like, shape (3, 3)
        Cartesian lattice vectors (rows are unit cell vectors in Å).
    condition : str
        Boolean expression using x, y, z (Å) and atom (symbol),
        e.g. "z < 3 and atom=='Si'".
    include : list of int, range, slice, or tuple, optional
        Atom indices (1-based), ranges, or slices to force include (sets True).
        Only affects listed atoms; others follow condition unless excluded.
    exclude : list of int, range, slice, or tuple, optional
        Atom indices (1-based), ranges, or slices to force exclude (sets False).
        Always overrides include and condition.

    Returns
    -------
    result : ndarray, shape (N, 6)
        [index(int,1-based), is_selected(bool), element_symbol, frac_x, frac_y, frac_z] per atom.
    """
    arr = np.array(positions_frac, dtype=object)
    symbols = arr[:, 0]
    frac = arr[:, 1:].astype(float)

    n_atoms = len(arr)
    def build_index_set(spec):
        s = set()
        for item in spec or []:
            if isinstance(item, int):
                s.add(item - 1)
            elif isinstance(item, range):
                s.update(i - 1 for i in item)
            elif isinstance(item, slice):
                start = item.start or 1
                stop = item.stop or n_atoms
                s.update(i for i in range(start - 1, stop))
            elif isinstance(item, tuple) and len(item) == 2:
                start, end = item
                s.update(i - 1 for i in range(start, end + 1))
            else:
                raise ValueError(
                    f"Invalid specifier '{item}'; use int, range, slice, or (start, end) tuple"
                )
        return s

    include_set = build_index_set(include)
    exclude_set = build_index_set(exclude)

    # Convert fractional to Cartesian
    cart = np.dot(frac, np.array(lattice_cart, dtype=float))

    output = []
    for idx, (atom, fcoords, ccoords) in enumerate(zip(symbols, frac, cart)):
        if idx in exclude_set:
            is_sel = False
        elif idx in include_set:
            is_sel = True
        else:
            x, y, z = ccoords
            try:
                is_sel = bool(eval(condition, {}, {'x': x, 'y': y, 'z': z, 'atom': atom}))
            except Exception as e:
                raise ValueError(
                    f"Error evaluating condition '{condition}' on atom index {idx+1} ({atom}): {e}"
                )
        py_fracs = [float(v) for v in fcoords]
        output.append([idx+1, is_sel, atom, *py_fracs])

    return np.array(output, dtype=object)


def select_atoms_by_plane(
    positions_frac,
    lattice_cart,
    plane,
    reference_position,
    tolerance=0.1
):
    """
    Select atoms lying within a slab around a specified plane.

    Parameters
    ----------
    positions_frac : array-like of shape (N, 4)
        Rows of [element_symbol, frac_x, frac_y, frac_z].
    lattice_cart : array-like of shape (3, 3)
        Cartesian cell vectors (each row is a lattice vector in Å).
    plane : array-like of length 3
        Normal vector (in Cartesian coordinates) defining the plane.
    reference_position : array-like of length 3
        A point (in Cartesian coordinates) known to lie on the plane.
    tolerance : float, optional
        Distance tolerance in Å. Atoms whose perpendicular distance to the plane
        is ≤ tolerance are selected (default = 0.0).

    Returns
    -------
    result : list of lists, shape (M, 6)
        Each entry corresponds to one atom and contains:
        [index (1-based), is_selected (bool), element_symbol,
         frac_x, frac_y, frac_z, distance]
        where `distance` is the signed perpendicular distance (Å) from the atom
        to the plane (positive on the side of the normal vector).
    """
    # Convert inputs
    arr = np.array(positions_frac, dtype=object)
    symbols = arr[:, 0]
    frac = arr[:, 1:].astype(float)
    lattice = np.array(lattice_cart, dtype=float)

    # Compute Cartesian positions of atoms
    cart = frac.dot(lattice)

    # Plane normal and reference point
    normal = np.array(plane, dtype=float)
    if np.allclose(normal, 0):
        raise ValueError("Plane normal vector must be nonzero.")
    # unit normal
    n_unit = normal / np.linalg.norm(normal)
    ref_pt = np.array(reference_position, dtype=float)

    # Compute signed distances
    # For each atom: d = dot(r_i - ref_pt, n_unit)
    deltas = cart - ref_pt
    dists = deltas.dot(n_unit)

    # Build result list
    result = []
    for i, (sym, fcoord, dist) in enumerate(zip(symbols, frac, dists)):
        sel = abs(dist) <= tolerance
        result.append([
            i + 1,
            bool(sel),
            sym,
            float(fcoord[0]),
            float(fcoord[1]),
            float(fcoord[2])
        ])

    return result


def select_atom_by_conditions(positions_frac, lattice_cart, criteria,
                              order=('z','y','x')):
    """
    Sequentially filter by (min|max) on axes in `order`, but if exactly one
    atom remains, return:
      - sel_frac:   1D array [species, fx, fy, fz]
      - sel_cart:   tuple (species, cx, cy, cz)
    Otherwise:
      - sel_frac:   2D object-array shape (M,4)
      - sel_cart:   2D object-array shape (M,4) with species + Cartesian coords
    """
    arr = np.asarray(positions_frac, dtype=object)
    frac = arr[:, 1:].astype(float)

    axis_map = {'x':0, 'y':1, 'z':2}
    idxs = np.arange(len(frac))

    for axis in order:
        i = axis_map[axis]
        crit = criteria[i]
        vals = frac[idxs, i]
        target = vals.min() if crit=='min' else vals.max()
        idxs = idxs[vals == target]
        if len(idxs) == 1:
            break

    # fractional selection (unchanged)
    sel_frac = arr[idxs]

    # compute Cartesian coords
    cart_coords = frac[idxs] @ np.asarray(lattice_cart, dtype=float)
    # species labels
    labels = arr[idxs, 0]

    # build labeled Cartesian array
    # cast coords to object so we can hstack with strings
    cart_obj = cart_coords.astype(object)
    sel_cart = np.hstack((labels.reshape(-1,1), cart_obj))

    # unwrap if single
    if sel_frac.shape[0] == 1:
        sel_frac = sel_frac[0]                    # 1D: [species, fx, fy, fz]
        # turn the single 1×4 array into a tuple (species, cx, cy, cz)
        sel_cart = tuple(sel_cart[0])

    return sel_frac, sel_cart


def select_atom_by_number(positions_frac, lattice_cart, atom_number, element=None):
    """
    Select the atom at position `atom_number` in positions_frac (1-based),
    optionally filtering by element, and return both fractional and Cartesian
    coords—each prefixed by the atom type.

    Returns
    -------
    sel_frac : np.ndarray, shape (4,), dtype object
        [symbol, frac_x, frac_y, frac_z]
    sel_cart : np.ndarray, shape (4,), dtype object
        [symbol, cart_x, cart_y, cart_z]
    """
    # Convert to object array
    arr = np.asarray(positions_frac, dtype=object)
    # Optionally filter by element type
    if element is not None:
        mask = arr[:, 0] == element
        filtered = arr[mask]
    else:
        filtered = arr

    N = len(filtered)
    if not (1 <= atom_number <= N):
        raise IndexError(f"atom_number {atom_number} out of range (1–{N})"
                         + (f" for element='{element}'" if element else ""))

    # Pick the requested row
    row = filtered[atom_number - 1]      # object array [sym, fx, fy, fz]
    symbol = row[0]
    frac_coords = row[1:].astype(float)  # [fx, fy, fz]

    # Compute Cartesian coords
    lattice = np.asarray(lattice_cart, dtype=float)
    cart_coords = frac_coords @ lattice  # [cx, cy, cz]

    # Build the outputs, as object arrays of length 4
    sel_frac = np.hstack(([symbol], frac_coords)).astype(object)
    sel_cart = np.hstack(([symbol], cart_coords)).astype(object)

    return tuple(sel_frac), tuple(sel_cart)


def selected_replace(data, element, replacewith, return_labelled=False):
    """
    For each row in `data` (list of lists), if the row contains the boolean True,
    return a new row where every element equal to `find` is replaced by `replace`.
    Rows without True are left untouched.
    """
    result = []
    for row in data:
        if any(cell is True for cell in row):
            # Replace matching entries in this row
            new_row = [replacewith if cell == element else cell for cell in row]
        else:
            # Leave the row as-is
            new_row = list(row)
        result.append(new_row)
    
    if not return_labelled:
        result = labelled_positions_frac_to_positions_frac(result)

    return result


def selected_translate(labelled_positions_frac: List[List[Union[int, bool, str, float]]],
                       cell: List[List[float]],
                       v: Tuple[float, float, float],
                       return_labelled: bool = False) -> List[List[Union[int, bool, str, float]]]:
    """
    Translate atoms marked True by a given displacement vector in Angstroms within the specified unit cell.

    Parameters:
    -----------
    labelled_positions_frac : List of atom entries [label, selected, element, x, y, z]
    cell : 3x3 list defining the unit cell vectors in Cartesian Angstroms
    v : (dx, dy, dz) displacement in Cartesian Angstroms
    return_labelled : If False, returns fractional positions array (drops boolean and labels via labelled_positions_frac_to_positions_frac)

    Returns:
    --------
    If return_labelled=True, returns a new labelled_positions_frac list with selected atoms moved by v
    Else returns only the fractional coordinates list (unit-cell scaled)
    """
    # Convert cell to numpy array and compute fractional displacement
    cell_matrix = np.array(cell, dtype=float)
    inv_cell = np.linalg.inv(cell_matrix)
    disp_frac = inv_cell.dot(np.array(v, dtype=float))

    # Prepare result copy
    result = [list(entry) for entry in labelled_positions_frac]
    for i, entry in enumerate(labelled_positions_frac):
        if entry[1] is True:
            # Add fractional displacement
            result[i][3] += float(disp_frac[0])
            result[i][4] += float(disp_frac[1])
            result[i][5] += float(disp_frac[2])
    if return_labelled:
        return result
    # Otherwise convert to plain fractional positions
    return labelled_positions_frac_to_positions_frac(result)


def selected_delete(data, return_labelled=False):
    """
    Return a new list of rows, omitting any row that contains the boolean True.
    """
    result = [row for row in data if not any(cell is True for cell in row)]

    if not return_labelled:
        result = labelled_positions_frac_to_positions_frac(result)

    return result


def selected_toggle_plane_selection(labelled_positions_frac: List[List[Union[int, bool, str, float]]], 
                            fast: str = 'y', 
                            slow: str = 'x', 
                            alternate: bool = False
                           ) -> Tuple[List[List[Union[int, bool, str, float]]],
                                       List[List[Union[int, bool, str, float]]]]:
    """
    Generate two selection patterns for atoms in a plane:

    1. A toggled pattern along the fast axis (striped or checkerboard).
    2. The inverse of that pattern, with originalFalse atoms always remaining False.

    Parameters
    ----------
    labelled_positions_frac : List of [label, selected, element, x, y, z]
    fast : 'x'|'y'|'z'  -- axis to scan fastest
    slow : 'x'|'y'|'z'  -- axis to scan slowest (must differ from fast)
    alternate : bool    -- if True, offset every other row (checkerboard), else stripes

    Returns
    -------
    result1, result2
      result1: list with toggled booleans for originally True atoms
      result2: list with inverted pattern of result1; atoms originally False remain False
    """
    # Axis to index mapping
    axis_map = {'x': 3, 'y': 4, 'z': 5}
    if fast not in axis_map or slow not in axis_map:
        raise ValueError("fast and slow must be one of 'x', 'y', or 'z'")
    if fast == slow:
        raise ValueError("fast and slow axes must differ")

    # Deep copies of input for both outputs
    result1 = [entry.copy() for entry in labelled_positions_frac]
    result2 = [entry.copy() for entry in labelled_positions_frac]

    # Collect indices and coords for originally True entries
    slow_idx = axis_map[slow]
    fast_idx = axis_map[fast]
    selected = []  # (orig_idx, slow_val, fast_val)
    for i, entry in enumerate(labelled_positions_frac):
        if entry[1] is True:
            selected.append((i, entry[slow_idx], entry[fast_idx]))

    if not selected:
        return result1, result2

    # Group by slow-axis value
    from collections import defaultdict
    groups = defaultdict(list)
    for idx, s_val, f_val in selected:
        groups[s_val].append((idx, f_val))

    # Sort slow coords for scanning order
    sorted_slow = sorted(groups.keys())

    # Build toggled pattern in result1
    for row_num, s_val in enumerate(sorted_slow):
        line = groups[s_val]
        # Sort along fast axis
        line_sorted = sorted(line, key=lambda x: x[1])  # (idx, fast_val)
        row_offset = row_num if alternate else 0
        for j, (orig_idx, _) in enumerate(line_sorted):
            if (j + row_offset) % 2 == 0:
                result1[orig_idx][1] = not result1[orig_idx][1]

    # Build inverse pattern in result2, but keep original False atoms False
    for i, entry in enumerate(labelled_positions_frac):
        if entry[1] is True:
            # invert the toggled result
            result2[i][1] = not result1[i][1]
        else:
            # ensure False remains False
            result2[i][1] = False

    return result1, result2


def update_labelled_positions_frac(
    labelled_positions_frac: List[List[Any]],
    positions_frac:         List[List[Any]]
) -> List[List[Any]]:
    """
    Replace the element type and (x,y,z) coords in labelled_positions_frac
    with those in positions_frac.

    Parameters
    ----------
    labelled_positions_frac : List of [id, flag, element, x, y, z]
    positions_frac          : List of [element, x, y, z]

    Returns
    -------
    new_labelled : a new list of the same shape as labelled_positions_frac,
                   where each row is [id, flag, new_element, new_x, new_y, new_z]

    Raises
    ------
    ValueError
        If the two input lists have different lengths.
    """
    if len(labelled_positions_frac) != len(positions_frac):
        raise ValueError(
            f"Length mismatch: "
            f"labelled_positions_frac has {len(labelled_positions_frac)} rows, "
            f"positions_frac has {len(positions_frac)} rows."
        )

    updated = []
    for (row_labelled, row_pos) in zip(labelled_positions_frac, positions_frac):
        id_, flag = row_labelled[0], row_labelled[1]
        elem, x, y, z = row_pos  # unpack element and coords
        updated.append([id_, flag, elem, x, y, z])

    return updated


def get_plane_lattice_vectors(
    labelled_positions_frac,
    lattice_cart
):
    """
    Determine two independent translational lattice vectors lying in the plane
    defined by a set of selected atoms.

    Parameters
    ----------
    labelled_positions_frac : array-like of shape (M, 7)
        Rows of [index, is_selected, element_symbol, frac_x, frac_y, frac_z, distance]
        as returned by `select_atoms_by_plane`.
    lattice_cart : array-like of shape (3, 3)
        Cartesian lattice vectors (rows are a, b, c vectors).

    Returns
    -------
    v1_cart, v2_cart : ndarray, shape (3,)
        Two independent Cartesian translation vectors that span the plane defined
        by the selected atoms.

    Raises
    ------
    ValueError
        If fewer than two atoms are selected or if two independent vectors cannot
        be found among the selected atoms.
    """
    arr = np.asarray(labelled_positions_frac, dtype=object)
    # Filter only selected atoms (is_selected == True)
    sel_mask = arr[:, 1].astype(bool)
    fracs = arr[sel_mask, 3:6].astype(float)

    if fracs.shape[0] < 2:
        raise ValueError("Need at least two selected atoms to define plane vectors.")

    lattice = np.asarray(lattice_cart, dtype=float)

    # Compute fractional differences with periodic wrapping
    deltas = []
    M = len(fracs)
    for i in range(M):
        for j in range(i+1, M):
            diff = fracs[j] - fracs[i]
            # wrap into [-0.5, +0.5]
            wrapped = diff - np.round(diff)
            if np.linalg.norm(wrapped) > 1e-8:
                deltas.append(wrapped)
    if not deltas:
        raise ValueError("No non-zero vector differences found among selected atoms.")
    deltas = np.array(deltas)

    # Sort differences by length and pick two independent vectors
    lengths = np.linalg.norm(deltas, axis=1)
    idx = np.argsort(lengths)
    deltas = deltas[idx]

    v_frac1 = deltas[0]
    for vec in deltas[1:]:
        if np.linalg.norm(np.cross(v_frac1, vec)) > 1e-6:
            v_frac2 = vec
            break
    else:
        raise ValueError("Unable to find two independent vectors in the plane.")

    # Convert fractional to Cartesian
    v1_cart = v_frac1.dot(lattice)
    v2_cart = v_frac2.dot(lattice)
    return v1_cart, v2_cart


def find_plane_value(positions_frac, lattice_cart, axis, criteria):
    """
    Determine the plane coordinate along an axis based on criteria.

    Parameters
    ----------
    positions_frac : array-like, shape (N, 4)
        Rows of [element_symbol, frac_x, frac_y, frac_z].
    lattice_cart : array-like, shape (3, 3)
        Cartesian lattice vectors (rows are unit cell vectors in Å).
    axis : {'x', 'y', 'z'}
        Axis perpendicular to the plane.
    criteria : {'minimum', 'maximum', 'centre'}
        'minimum' returns the smallest coordinate, 'maximum' the largest,
        'centre' the midpoint between min and max.

    Returns
    -------
    plane_cart : float
        Cartesian coordinate in Å of the plane.
    plane_frac : float
        Fractional coordinate (0–1) along the axis of the plane.
    """
    arr = np.array(positions_frac, dtype=object)
    frac = arr[:, 1:].astype(float)
    lattice = np.array(lattice_cart, dtype=float)

    # Map axis to index
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    try:
        ai = axis_map[axis.lower()]
    except KeyError:
        raise ValueError("Axis must be one of 'x', 'y', or 'z'")

    # Fractional coordinates along axis
    frac_coords = frac[:, ai]
    if criteria == 'minimum':
        plane_frac = frac_coords.min()
    elif criteria == 'maximum':
        plane_frac = frac_coords.max()
    elif criteria == 'centre':
        plane_frac = (frac_coords.min() + frac_coords.max()) / 2.0
    else:
        raise ValueError("Criteria must be 'minimum', 'maximum', or 'centre'")

    # Convert to Cartesian
    cell_vec = lattice[ai]
    plane_cart = plane_frac * np.linalg.norm(cell_vec)

    return float(plane_cart), float(plane_frac)


# ============================================================================
#endregion Manipulate arrays, lists, coordinates, and cells. 
# ============================================================================


# ============================================================================
#region Generate APOLLO job submission scripts
# ============================================================================

def write_job_script(
    path,
    filename,
    wall_time='24:00:00',
    queue_name='A_192T_1024G.q',
    available_cores=192,
    available_memory='1024G',
    threads=1,
    safety_factor=0.98,
    display_file=False
):
    """
    Write a job submission script for SGE with hybrid MPI+OpenMP threading,
    allocating a conservative estimate of memory per slot by applying a safety margin.

    Args:
        path (str or Path): Directory where the .job file will be created.
        filename (str): Base name for the job file (no extension).
        wall_time (str): Wallclock time in format HH:MM:SS. Default '24:00:00'.
        queue_name (str): Name of the SGE queue. Default 'A_192T_1024G.q'.
        available_cores (int): Total CPU cores available on the queue. Default 192.
        available_memory (str or int): Total memory available on the queue
            (e.g., '1024G', '1048576M', or integer GB). Default '1024G'.
        threads (int): OMP threads per MPI rank. Default 1.
        safety_factor (float): Fraction of total memory to allocate (e.g., 0.98 for 2% margin).
            Must be between 0 and 1. Default 0.98.
        display_file (bool): If True, print the script contents after writing.

    Returns:
        Path: Path to the written .job file.

    Raises:
        ValueError: If threads do not divide available_cores or safety_factor out of range
                    or memory string format is invalid.
    """
    # Validate threads and safety
    if available_cores % threads != 0:
        raise ValueError(f"threads ({threads}) must divide available_cores ({available_cores}) evenly.")
    if not (0 < safety_factor <= 1):
        raise ValueError("safety_factor must be between 0 (exclusive) and 1 (inclusive).")

    n_ranks = available_cores // threads

    # Parse available_memory into megabytes
    if isinstance(available_memory, int):
        total_mem_mb = available_memory * 1024
    else:
        mem_str = str(available_memory).strip().upper()
        if mem_str.endswith('G'):
            total_mem_mb = int(mem_str[:-1]) * 1024
        elif mem_str.endswith('M'):
            total_mem_mb = int(mem_str[:-1])
        else:
            raise ValueError("available_memory must be an integer or end with 'G' or 'M'.")

    # Apply safety margin and compute per-slot memory
    effective_mem_mb = int(total_mem_mb * safety_factor)
    mem_per_slot_mb = effective_mem_mb // available_cores
    mem_per_slot = f"{mem_per_slot_mb}M"

    # Ensure output directory exists
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build job file path
    job_file = out_dir / f"{filename}.job"

    # Generate script content
    content = f"""#!/bin/bash
#$ -N {filename}                         # Job name
#$ -q {queue_name}                    # Queue name
#$ -l h_rt={wall_time}                     # Wall-clock time limit
#$ -pe ompi-local {available_cores}                   # Request {available_cores} CPU slots
#$ -l vf={mem_per_slot}                          # Memory per core slot (~{mem_per_slot}/core)
#$ -V                                   # Export environment variables
#$ -cwd                                 # Run in current working directory
#$ -j y                                 # Join stdout and stderr
#$ -o {filename}.apollo.log              # Combined log file
#$ -S /bin/bash                         # Use bash shell

# Set OpenMP threads
export OMP_NUM_THREADS={threads}
echo "Threading set to: OMP_NUM_THREADS={threads}"

# Path to personal modules
module use /hpc/srs/local/privatemodules/
module purge
module load CASTEP-24
module load modules sge

echo "The following modules are loaded"
module list

# Activate conda environment
source /hpc/srs/local/miniconda3/etc/profile.d/conda.sh
conda activate apollo_castep

# Diagnostics
echo "Allocated slots: $NSLOTS"
echo "Reserved memory per slot: $VF"
echo "Host: $(hostname)"
echo "Python executable: $(which python)"
echo "Working directory: $(pwd)"

# Run the CASTEP calculation
echo "Running CASTEP calculation with {n_ranks} MPI ranks and {threads} OpenMP threads per rank."
mpirun -np {n_ranks} castep.mpi {filename}
"""

    # Write to disk
    with open(job_file, 'w') as f:
        f.write(content)

    # Optionally display to console
    if display_file:
        print(content)

    return job_file


# ============================================================================
#endregion Generate APOLLO job submission scripts
# ============================================================================


# ============================================================================
#region MACRO like functiond
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
            cell = get_lattice_parameters(castep_path)
            general = get_summary_parameters(castep_path)
            enthalpy = get_LBFGS_final_enthalpy(castep_path)
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


def optimisation_summary_macro_1(castep_paths):
    for castep_path in castep_paths:
        # Header and error information
        print_filename(castep_path)
        #ct.extract_warnings(castep_path,verbose=True)

        # Energy convergence
        convergence = get_LBFGS_energies(castep_path)
        final_enthalpy = get_LBFGS_final_enthalpy(castep_path)
        print('Final enthalpy = {} eV.'.format(final_enthalpy))
        plot_energy_vs_iteration(convergence, title=castep_path.stem+' '+str(final_enthalpy),figsize=(5,2))
        
        # Unit cell parameters
        cell = get_calculation_parameters(castep_path,a0=3.8668346, vac=15.0)
        cell_df = pd.DataFrame(cell.items(), columns=["Cell parameters", "Value"])
        display(cell_df) 

        # Show structure
        atoms = fractional_coords_from_castep(castep_path)
        view = view_structure(atoms,show_structure=False)
        display(view)
        img = view.render_image(frame=None, factor=4, antialias=True, trim=False, transparent=False)
        display(img)

# ============================================================================
#endregion Generate APOLLO job submission scripts
# ============================================================================