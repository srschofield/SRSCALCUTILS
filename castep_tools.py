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


def extract_summary_parameters(castep_path):
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

import re

def extract_LBFGS_final_enthalpy(castep_path):
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


# ============================================================================
#  Structure information
# ============================================================================

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



def extract_final_fractional_positions(castep_path):
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

# ============================================================================
#  Macro like functions doing multiple things
# ============================================================================
def optimisation_summaries(castep_paths):
    for castep_path in castep_paths:
        # Header and error information
        print_file_info(castep_path)
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

def write_block_lattice_cart(a=3.8668346,b=3.8668346,c=5.4685299,nx=1,ny=1,nz=1):
    """
    Write a block lattice in Cartesian coordinates.
    """
    lattice = np.diag([nx * a, ny * b, nz * c])

    lines = ["%BLOCK lattice_cart", "   ANG"]
    for row in lattice:
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


def write_fractional_bulk_coords(
    nx=1,
    ny=1,
    nz=1,
    atom="Si",
    unit_cell=None
):
    """
    Generate a fractional-coordinate supercell block.

    Parameters
    ----------
    nx, ny, nz : int
        Number of repetitions along x, y, z.
    atom : str
        Element label to prepend to each line.
    unit_cell : array-like of shape (M,3), optional
        Fractional positions of the M atoms in the unit cell.
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
    # default 4‐atom Si cell if none provided
    if unit_cell is None:
        unit_cell = np.array([
            [0.0,  0.0,  0.0],
            [0.5,  0.0,  0.25],
            [0.5,  0.5,  0.5],
            [0.0,  0.5,  0.75],
        ])
    else:
        # ensure it’s a NumPy array
        unit_cell = np.asarray(unit_cell, dtype=float)
        if unit_cell.ndim != 2 or unit_cell.shape[1] != 3:
            raise ValueError("unit_cell must be an array-like of shape (M, 3)")

    super_cell = []
    # tile the basis over each (i,j,k) cell
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for pos in unit_cell:
                    # shift then normalize into supercell fractional coords
                    new_pos = [
                        (pos[0] + i) / nx,
                        (pos[1] + j) / ny,
                        (pos[2] + k) / nz
                    ]
                    super_cell.append(new_pos)

    # build the text block
    lines = ["%BLOCK positions_frac"]
    for row in super_cell:
        fields = [f"{val:16.10f}" for val in row]
        lines.append("   " + atom + "   " + "".join(fields))
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

def write_cell_file(atom, nx, ny, nz, 
                    a=3.8668346,
                    b=3.8668346,
                    c=5.4685299,
                    unit_cell=None,
                    constraints=None,
                    fix_all_ions=True,
                    symmetry_generate=True,
                    symmetry_tol=None,
                    kpoints_mp_grid=None,
                    title=None,
                    filename='crystal', 
                    path=".", 
                    display_file=False):
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
    # 1) Build the three text‐blocks
    lattice_block = write_block_lattice_cart(a=a, b=b, c=c, nx=nx, ny=ny, nz=nz)
    constraint_block = write_cell_constraints(constraints=constraints)
    frac_block = write_fractional_bulk_coords(nx=nx, ny=ny, nz=nz, atom=atom,unit_cell=unit_cell)
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
    
    if title:
        title_block = f"!TITLE: {title}"
    else:
        title_block = ""

    # 2) Concatenate with blank lines between
    full_text = "\n\n".join([
        title_block,
        lattice_block, 
        constraint_block, 
        frac_block,
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
