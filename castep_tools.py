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
#  Basic information
# ============================================================================

def extract_warnings(castep_path, verbose=True):
    """
    Extracts and prints WARNING blocks from a .castep file.
    
    Parameters:
        castep_path (str or Path): Path to the .castep file.
        verbose (bool): If True, print full blocks until the next blank line.
                        If False, only print the matching WARNING line.
    """
    from pathlib import Path

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
