
from importlib.resources import files
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from pyagadir.utils import is_valid_peptide_sequence, is_valid_index


# get params
datapath = files('pyagadir.data')

# load energy contributions for intrinsic propensities, capping, etc.
table_1_lacroix = pd.read_csv(
    datapath.joinpath('table_1_lacroix.tsv'),
    index_col='AA',
    sep='\t',
).astype(float)

# load energy contributions between amino acids and the helix macrodipole, focusing on the C-terminal
table3a = pd.read_csv(
    datapath.joinpath('table3a.csv'),
    index_col='AA',
).astype(float)
table3a.columns = table3a.columns.astype(int)

# load energy contributions between amino acids and the helix macrodipole, focusing on the N-terminal
table3b = pd.read_csv(
    datapath.joinpath('table3b.csv'),
    index_col='AA',
).astype(float)
table3b.columns = table3b.columns.astype(int)

# load energy contributions for interactions between i and i+3
table4a = pd.read_csv(
    datapath.joinpath('table4a.csv'),
    index_col='index',
).astype(float)

# load energy contributions for interactions between i and i+4
table4b = pd.read_csv(
    datapath.joinpath('table4b.csv'),
    index_col='index',
).astype(float)


def get_helix(pept: str, i: int, j:int) -> str:
    """
    Get the helix region of a peptide sequence.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.
    
    Returns:
        str: The helix region of the peptide sequence.
    """
    is_valid_peptide_sequence(pept)
    is_valid_index(pept, i, j)

    return pept[i:i+j]


def get_dG_Int(pept: str, i: int, j: int, pH: float = 7.0) -> np.ndarray:
    """
    Get the intrinsic free energy contributions for a sequence. 
    The first and last residues are considered to be the caps and 
    do not contribute to the intrinsic free energy.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.
        pH (float): The pH value. Default is 7.0.

    Returns:
        np.ndarray: The intrinsic free energy contributions for each amino acid in the sequence.
    """
    is_valid_peptide_sequence(pept)
    is_valid_index(pept, i, j)

    # get the helix
    helix = get_helix(pept, i, j)

    # initialize energy array
    energy = np.zeros(len(helix))

    # iterate over the helix and get the intrinsic energy for each residue
    for idx, AA in enumerate(helix):

        # skip caps 
        if idx in [0, len(helix)-1]:
            continue

        # get the residue intrinsic energy
        if idx == 1:
            energy[idx] = table_1_lacroix.loc[AA, 'N1']

        elif idx == 2:
            energy[idx] = table_1_lacroix.loc[AA, 'N2']

        elif idx == 3:
            energy[idx] = table_1_lacroix.loc[AA, 'N3']

        elif idx == 4:
            energy[idx] = table_1_lacroix.loc[AA, 'N4']

        else:
            if AA not in ['C', 'D', 'E', 'H', 'K', 'R', 'Y']:
                energy[idx] = table_1_lacroix.loc[AA, 'Ncen']
            else:
                ## TODO decide when to pick Ncen or Neutral depending on the pH
                energy[idx] = table_1_lacroix.loc[AA, 'Ncen'] # what avout Neutral?

    return energy


def get_dG_Ncap(pept: str, i: int, j: int) -> np.ndarray:
    """
    Get the free energy contribution for N-terminal capping.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.

    Returns:
        np.ndarray: The free energy contribution.
    """
    is_valid_peptide_sequence(pept)
    is_valid_index(pept, i, j)

    # get the helix
    helix = get_helix(pept, i, j)

    # fix the blocking group names to match the table
    AA = helix[0]
    if AA in ['Z', 'X']:
        AA = 'Ac'

    energy = np.zeros(len(helix))

    # Nc-4 	N-cap values when there is a Pro at position N1 and Glu, Asp or Gln at position N3.  
    if helix[1] == 'P' and helix[3] in ['E', 'D', 'Q']:
        energy[0] = table_1_lacroix.loc[AA, 'Nc-4']
    
    # Nc-3 	N-cap values when there is a Glu, Asp or Gln at position N3.
    elif helix[3] in ['E', 'D', 'Q']:
        energy[0] = table_1_lacroix.loc[AA, 'Nc-3']

    # Nc-2 	N-cap values when there is a Pro at position N1.
    elif helix[1] == 'P':
        energy[0] = table_1_lacroix.loc[AA, 'Nc-2']

    # Nc-1 	Normal N-cap values.
    else:
        energy[0] = table_1_lacroix.loc[AA, 'Nc-1']

    return energy


def get_dG_Ccap(pept: str, i: int, j: int) -> np.ndarray:
    """
    Get the free energy contribution for N-terminal capping.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.

    Returns:
        np.ndarray: The free energy contribution.
    """
    is_valid_peptide_sequence(pept)
    is_valid_index(pept, i, j)

    # get the helix
    helix = get_helix(pept, i, j)

    # fix the blocking group names to match the table
    AA = helix[-1]
    if AA == 'B':
        AA = 'Am'
    
    energy = np.zeros(len(helix))

    # Cc-2 	C-cap values when there is a Pro residue at position C'
    c_prime_idx = i+j
    if (len(pept) > c_prime_idx) and (pept[c_prime_idx] == 'P'):
        energy[-1] = table_1_lacroix.loc[AA, 'Cc-2']

    # Cc-1 	Normal C-cap values
    else:
        energy[-1] = table_1_lacroix.loc[AA, 'Cc-1']

    return energy


def get_dG_Hbond(seq: str) -> float:
    """
    Get the free energy contribution for hydrogen bonding for a sequence.

    Args:
        seq (str): The protein sequence.

    Returns:
        float: The total free energy contribution for hydrogen bonding in the sequence.
    """
    is_valid_peptide_sequence(seq)

    # The first 4 helical amino acids are considered to have zero net enthalpy 
    # since they are nucleating residues and caps don't count, 
    # for a total of 6.
    j = len(seq)
    energy = -0.775 * max((j - 6), 0)

    return energy


def get_dG_i1(seq: str) -> np.ndarray:
    """
    Get the free energy contribution for interaction between each AAi and AAi+1 in the sequence.

    Args:
        seq (str): The protein sequence.

    Returns:
        np.ndarray: The free energy contributions for each interaction.
    """
    is_valid_peptide_sequence(seq)

    energy = np.zeros(len(seq))
    for i in range(len(seq) - 1):
        AAi = seq[i]
        AAi1 = seq[i + 1]
        charge = 1
        for AA in [AAi, AAi1]:
            if AA in set(['R', 'H', 'K']):
                charge *= 1
            elif AA in set(['D', 'E']):
                charge *= -1
            else:
                charge = 0
                break
        energy[i] = 0.05 * charge if charge != 0 else 0.0
    return energy


def get_dG_i3(seq: str) -> np.ndarray:
    """
    Get the free energy contribution for interaction between each AAi and AAi+3 in the sequence.

    Args:
        seq (str): The protein sequence.

    Returns:
        np.ndarray: The free energy contributions for each interaction.
    """
    is_valid_peptide_sequence(seq)   

    energy = np.zeros(len(seq))
    for i in range(len(seq) - 3):
        AAi = seq[i]
        AAi3 = seq[i + 3]
        energy[i] = table4a.loc[AAi, AAi3]
    return energy


def get_dG_i4(seq: str) -> np.ndarray:
    """
    Get the free energy contribution for interaction between each AAi and AAi+4 in the sequence.

    Args:
        seq (str): The protein sequence.

    Returns:
        np.ndarray: The free energy contributions for each interaction.
    """
    is_valid_peptide_sequence(seq)

    energy = np.zeros(len(seq))
    for i in range(len(seq) - 4):
        AAi = seq[i]
        AAi4 = seq[i + 4]
        energy[i] = table4b.loc[AAi, AAi4]
    return energy


def get_dG_dipole(seq: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the dipole free energy contribution.
    The nomenclature is that of Richardson & Richardson (1988),
    which is different from the one used in the AGADIR paper.
    Richardson considers the first and last helical residues as the caps.

    Args:
        seq (str): The amino acid sequence.

    Returns:
        tuple[np.ndarray, np.ndarray]: The dipole free energy contributions for N-terminal and C-terminal.
    """
    is_valid_peptide_sequence(seq)
    
    N = len(seq)
    dG_N_dipole = np.zeros(N)
    dG_C_dipole = np.zeros(N)

    # N-term dipole contributions
    for i in range(0, min(N, 10)):
        dG_N_dipole[i] = table3a.loc[seq[i], i]

    # C-term dipole contributions
    seq_inv = seq[::-1]
    for i in range(0, min(N, 10)):
        dG_C_dipole[i] = table3b.loc[seq_inv[i], i*-1]
    dG_C_dipole = dG_C_dipole[::-1]

    return dG_N_dipole, dG_C_dipole
