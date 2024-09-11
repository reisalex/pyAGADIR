
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

# load the hydrophobic staple motif energy contributions
table_2_lacroix = pd.read_csv(
    datapath.joinpath('table_2_lacroix.tsv'),
    index_col='AA',
    sep='\t',
).astype(float)

# load the schellman motif energy contributions
table_3_lacroix = pd.read_csv(
    datapath.joinpath('table_3_lacroix.tsv'),
    index_col='AA',
    sep='\t',
).astype(float)

# load energy contributions for interactions between i and i+3
table_4a_lacroix = pd.read_csv(
    datapath.joinpath('table_4a_lacroix.tsv'),
    index_col='AA',
    sep='\t',
).astype(float)

# load energy contributions for interactions between i and i+4
table_4b_lacroix = pd.read_csv(
    datapath.joinpath('table_4b_lacroix.tsv'),
    index_col='AA',
    sep='\t',
).astype(float)


# # load energy contributions between amino acids and the helix macrodipole, focusing on the C-terminal
# table3a = pd.read_csv(
#     datapath.joinpath('table3a.csv'),
#     index_col='AA',
# ).astype(float)
# table3a.columns = table3a.columns.astype(int)

# # load energy contributions between amino acids and the helix macrodipole, focusing on the N-terminal
# table3b = pd.read_csv(
#     datapath.joinpath('table3b.csv'),
#     index_col='AA',
# ).astype(float)
# table3b.columns = table3b.columns.astype(int)


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

    # TODO ensure that the code below is correct

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


def get_dG_staple(pept: str, i: int, j: int) -> float:
    """
    Get the free energy contribution for the hydrophobic staple motif.
    The hydrophobic interaction is between the N' and N4 residues of the helix.
    See https://doi.org/10.1038/nsb0595-380 for more details.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.

    Returns:
        float: The free energy contribution.
    """
    is_valid_peptide_sequence(pept)
    is_valid_index(pept, i, j)

    # get the helix
    helix = get_helix(pept, i, j)

    energy = np.zeros(len(helix))

    # get the amino acids governing the staple motif
    N_prime_AA = pept[i-1]
    Ncap_AA = helix[0]
    N3_AA = helix[3]
    N4_AA = helix[4]
    energy = 0.0

    # staple motif requires the N' residue before the Ncap, so the first residue of the helix cannot be the first residue of the peptide
    if i == 0:
        return energy

    # TODO: verify that the code below is correct 

    # The hydrophobic staple motif is only considered whenever the N-cap residue is Asn, Asp, Ser, Pro or Thr. 
    if Ncap_AA in ['N', 'D', 'S', 'P', 'T']:
        energy = table_2_lacroix.loc[N_prime_AA, N4_AA]

        # whenever the N-cap residue is Asn, Asp, Ser, or Thr and the N3 residue is Glu, Asp or Gln, multiply by 1.0
        if Ncap_AA in ['N', 'D', 'S', 'T'] and N3_AA in ['E', 'D', 'Q']:
            print('staple case i')
            energy *= 1.0

        # whenever the N-cap residue is Asp or Asn and the N3 residue is Ser or Thr
        elif Ncap_AA in ['N', 'D'] and N3_AA in ['S', 'T']:
            print('staple case ii')
            energy *= 1.0

        # other cases they are multiplied by 0.5
        else:
            print('staple case iii')
            energy *= 0.5

    else:
        print('no staple motif')
        
    return energy


def get_dG_schellman(pept: str, i: int, j: int) -> float:
    """
    Get the free energy contribution for the Schellman motif.
    The Schellman motif is only considered whenever Gly is the C-cap residue,
    where the interaction happens between the C' and C3 residues of the helix.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.

    Returns:
        float: The free energy contribution.
    """
    is_valid_peptide_sequence(pept)
    is_valid_index(pept, i, j)

    # get the helix
    helix = get_helix(pept, i, j)
    energy = 0.0

    # TODO verify that the code below is correct

    # C-cap residue has to be Gly
    if helix[-1] != 'G':
        print('no G cap for schellman')
        return energy

    # there has to be a C' residue after the helix
    if i+j >= len(pept):
        print('no C prime for schellman')
        return energy
    
    # get the amino acids governing the Schellman motif and extract the energy
    print('detected schellman case')
    C3_AA = helix[3]
    C_prime_AA = pept[i+j]
    energy = table_3_lacroix.loc[C3_AA, C_prime_AA] / 100

    return energy


def get_dG_Hbond(pept: str, i: int, j: int) -> float:
    """
    Get the free energy contribution for hydrogen bonding for a sequence.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.

    Returns:
        float: The total free energy contribution for hydrogen bonding in the sequence.
    """
    is_valid_peptide_sequence(pept)
    is_valid_index(pept, i, j)

    # The first 4 helical amino acids are considered to have zero net enthalpy 
    # since they are nucleating residues and caps don't count, 
    # for a total of 6.
    energy = -0.775 * max((j - 6), 0)

    return energy


def get_dG_i1(pept: str, i: int, j: int) -> np.ndarray:
    """
    Get the free energy contribution for interaction between each AAi and AAi+1 in the sequence.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.

    Returns:
        np.ndarray: The free energy contributions for each interaction.
    """
    is_valid_peptide_sequence(pept)
    is_valid_index(pept, i, j)

    # NOTE: the this is the "old" code from the other Agadir implementation, have not changed it yet. Unclear whether I should.

    energy = np.zeros(len(pept))
    for i in range(len(pept) - 1):
        AAi = pept[i]
        AAi1 = pept[i + 1]
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


def get_dG_i3(pept: str, i: int, j: int) -> np.ndarray:
    """
    Get the free energy contribution for interaction between each AAi and AAi+3 in the sequence.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.

    Returns:
        np.ndarray: The free energy contributions for each interaction.
    """
    is_valid_peptide_sequence(pept)
    is_valid_index(pept, i, j)

    energy = np.zeros(len(pept))

    # Get interaction free energies between non-charged residues
    for i in range(len(pept) - 3):
        AAi = pept[i]
        AAi3 = pept[i + 3]
        energy[i] = table_4a_lacroix.loc[AAi, AAi3] / 100

        # TODO: I have to add values from table 5 of the lacroix paper, depending on ionization state of the residues. But how to do this?
        # "The interaction free energies correspond to those between non-charged residues, or in the case of two residues that can be charged 
        # to those cases in which at least one of the two is non-charged (the interaction is scaled according to the population of charged and 
        # neutral forms of the participating amino acids)."

    return energy


def get_dG_i4(pept: str, i: int, j: int) -> np.ndarray:
    """
    Get the free energy contribution for interaction between each AAi and AAi+4 in the sequence.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.

    Returns:
        np.ndarray: The free energy contributions for each interaction.
    """
    is_valid_peptide_sequence(pept)
    is_valid_index(pept, i, j)

    energy = np.zeros(len(pept))

    # Get interaction free energies between non-charged residues
    for i in range(len(pept) - 4):
        AAi = pept[i]
        AAi4 = pept[i + 4]
        energy[i] = table_4b_lacroix.loc[AAi, AAi4] / 100

        # TODO: I have to add values from table 5 of the lacroix paper, depending on ionization state of the residues. But how to do this?
        # "The interaction free energies correspond to those between non-charged residues, or in the case of two residues that can be charged 
        # to those cases in which at least one of the two is non-charged (the interaction is scaled according to the population of charged and 
        # neutral forms of the participating amino acids)."

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
