
from importlib.resources import files
import math

import numpy as np
import pandas as pd

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
    helix = get_helix(pept, i, j)

    # fix the blocking group names to match the table
    Ncap_AA = helix[0]
    if Ncap_AA in ['Z', 'X']:
        Ncap_AA = 'Ac'

    energy = np.zeros(len(helix))

    # Nc-4 	N-cap values when there is a Pro at position N1 and Glu, Asp or Gln at position N3.
    N1_AA = helix[1]
    N3_AA = helix[3]
    if N1_AA == 'P' and N3_AA in ['E', 'D', 'Q']:
        energy[0] = table_1_lacroix.loc[Ncap_AA, 'Nc-4']
    
    # Nc-3 	N-cap values when there is a Glu, Asp or Gln at position N3.
    elif N3_AA in ['E', 'D', 'Q']:
        energy[0] = table_1_lacroix.loc[Ncap_AA, 'Nc-3']

    # Nc-2 	N-cap values when there is a Pro at position N1.
    elif N1_AA == 'P':
        energy[0] = table_1_lacroix.loc[Ncap_AA, 'Nc-2']

    # Nc-1 	Normal N-cap values.
    else:
        energy[0] = table_1_lacroix.loc[Ncap_AA, 'Nc-1']

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
    helix = get_helix(pept, i, j)

    # fix the blocking group names to match the table
    Ccap_AA = helix[-1]
    if Ccap_AA == 'B':
        Ccap_AA = 'Am'
    
    energy = np.zeros(len(helix))

    # Cc-2 	C-cap values when there is a Pro residue at position C'
    c_prime_idx = i+j
    if (len(pept) > c_prime_idx) and (pept[c_prime_idx] == 'P'):
        energy[-1] = table_1_lacroix.loc[Ccap_AA, 'Cc-2']

    # Cc-1 	Normal C-cap values
    else:
        energy[-1] = table_1_lacroix.loc[Ccap_AA, 'Cc-1']

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
    energy = -0.895 * max((j - 6), 0)

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
    helix = get_helix(pept, i, j)

    # NOTE: the this is the "old" code from the other Agadir implementation, have not changed it yet. Unclear whether I should.

    energy = np.zeros(len(helix))
    for idx in range(len(helix) - 1):
        AAi = helix[idx]
        AAi1 = helix[idx + 1]
        charge = 1
        for AA in [AAi, AAi1]:
            if AA in set(['R', 'H', 'K']):
                charge *= 1
            elif AA in set(['D', 'E']):
                charge *= -1
            else:
                charge = 0
                break
        energy[idx] = 0.05 * charge if charge != 0 else 0.0
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
    helix = get_helix(pept, i, j)

    energy = np.zeros(len(helix))

    # Get interaction free energies between non-charged residues
    for idx in range(len(helix) - 3):
        AAi = helix[idx]
        AAi3 = helix[idx + 3]
        energy[idx] = table_4a_lacroix.loc[AAi, AAi3] / 100

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
    helix = get_helix(pept, i, j)

    energy = np.zeros(len(helix))

    # Get interaction free energies between non-charged residues
    for idx in range(len(helix) - 4):
        AAi = helix[idx]
        AAi4 = helix[idx + 4]
        energy[idx] = table_4b_lacroix.loc[AAi, AAi4] / 100

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


def acidic_residue_ionization(pH, pKa):
    """Degree of ionization indicates the fraction of molecules that
    are protonated (neutral) vs. deprotonated (negatively charged).
    Uses the Henderson-Hasselbalch equation to calculate the degree of ionization.
    """
    q_acid = 1 / (1 + 10**(pH - pKa))
    return q_acid


def basic_residue_ionization(pH, pKa):
    """Degree of ionization indicates the fraction of molecules that
    are protonated (positively charged) vs. deprotonated (neutral).
    Uses the Henderson-Hasselbalch equation to calculate the degree of ionization.
    """
    q_base = 1 / (1 + 10**(pKa - pH))
    return q_base


def calculate_r(N):
    """Function to calculate the distance r from the terminal to the helix
    where N is the number of residues between the terminal and the helix.
    """
    #TODO: find reference for this. 
    r = 0.1 + (N+1) * 2
    return r


def debye_huckel_simple(ionic_strength, distance_r):
    """Function to calculate the Debye-Huckel screening factor for electrostatic interactions
    given the ionic strength and distance r from the helix. Uses the simplified Debye-Huckel equation.
    which is valid for low ionic strength conditions in dilute solutions.
    """
    # Calculate the screening parameter kappa
    kappa = 0.329 * math.sqrt(ionic_strength)
    # Calculate the screening factor
    screening_factor = math.exp(-kappa * distance_r)
    return screening_factor


def debye_huckel_full(distance_r, ionic_strength):
    # Constants
    epsilon_0 = 8.854e-12  # Permittivity of free space in C^2/(Nm^2)
    epsilon_r = 80  # Relative permittivity (dielectric constant) of water
    N_A = 6.022e23  # Avogadro's number in mol^-1
    e = 1.602e-19  # Elementary charge in Coulombs
    k_B = 1.38e-23  # Boltzmann constant in J/K
    T = 273  # Temperature in Kelvin
    # Convert distance from Ångströms to meters
    distance_r = distance_r * 1e-10
    # Calculate Debye screening parameter kappa
    kappa = math.sqrt((2 * N_A * e**2 * ionic_strength) / (epsilon_0 * epsilon_r * k_B * T))
    # Calculate the screening factor e^(-kappa * r)
    screening_factor = math.exp(-kappa * distance_r)
    return screening_factor


def calculate_interaction_energy(q, mu_helix, distance_r, screening_factor):
    """Function to calculate the interaction energy between charged termini and the helix dipole
    q: degree of ionization (fraction of charged molecules)
    mu_helix: helix dipole moment
    distance_r: distance from the terminal to the helix
    screening_factor: Debye-Huckel screening factor
    """
    energy = ((q * mu_helix) / distance_r) * screening_factor
    return energy


def get_dG_terminals(pept: str, i: int, j: int, ionic_strength: float, pH: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the interaction energy for each residue with the N and C terminals
    """
    mu_helix = 0.5
    qKaN = 3.75 # TODO: Add values for each aa
    qKaC = 7.88 # TODO: Add values for each aa
    helix = get_helix(pept, i, j)
    N_term = np.zeros(len(helix))
    C_term = np.zeros(len(helix))
    # N terminal
    residue = helix[0]
    q_acid = acidic_residue_ionization(pH, qKaN)
    q = 1 - q_acid # TODO: ?
    distance_r = calculate_r(i) # Distance to N terminal
    screening_factor = debye_huckel_full(distance_r, ionic_strength)
    N_term_energy = calculate_interaction_energy(q, mu_helix, distance_r, screening_factor)
    N_term[0] = N_term_energy
    # C terminal
    residue = helix[-1]
    q_base = basic_residue_ionization(pH, qKaC)
    q = 1 - q_base # TODO: ?
    distance_r = calculate_r(len(pept)-(i+j)) # Distance to C terminal
    screening_factor = debye_huckel_full(distance_r, ionic_strength)
    C_term_energy = calculate_interaction_energy(q, mu_helix, distance_r, screening_factor)
    C_term[-1] = C_term_energy
    return N_term, C_term
   

