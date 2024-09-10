
from importlib.resources import files
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

# get params
datapath = files('pyagadir.data')

# load energy contributions for intrinsic propensities, capping, etc.
table2 = pd.read_csv(
    datapath.joinpath('table2.csv'),
    index_col='AA',
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


class ModelResult(object):
    """
    Class representing the result of a model.
    """

    def __init__(self, seq: str) -> None:
        """
        Initialize the ModelResult object.

        Args:
            seq (str): The peptide sequence.
        """
        self.seq: str = seq
        n: int = len(seq)
        self.dG_dict_mat: List[List[None]] = [None for j in range(5)] + [[None for _ in range(0, n - j)] for j in range(5, n)] # helix length is at least 6 but we zero-index
        self.K_tot: float = 0.0
        self.K_tot_array: np.ndarray = np.zeros(len(seq))
        self.Z: float = 0.0
        self.Z_array: np.ndarray = np.zeros(len(seq))
        self.helical_propensity: np.ndarray = None
        self.percent_helix: float = None

    def __repr__(self) -> str:
        """
        Return a string representation of the helical propensity.

        Returns:
            str: The helical propensity.
        """
        return str(self.helical_propensity)
    
    def get_sequence(self) -> str:
        """
        Get the peptide sequence.

        Returns:
            str: The peptide sequence.
        """
        return self.seq

    def get_helical_propensity(self) -> np.ndarray:
        """
        Get the helical propensity.

        Returns:
            np.ndarray: The helical propensity for each amino acid.
        """
        return self.helical_propensity

    def get_percent_helix(self) -> float:
        """
        Get the percentage of helix.

        Returns:
            float: The percentage of helix for the peptide.
        """
        return self.percent_helix


def is_valid_amino_acid_sequence(seq: str) -> None:
    """
    Validate that the input is a valid amino acid sequence.

    Args:
        seq (str): The input sequence.

    Raises:
        TypeError: If the input is not a string.
        ValueError: If the sequence contains invalid amino acids.
    """
    if not isinstance(seq, str):
        raise TypeError("Input must be a string.")
    
    if not len(seq) >= 6:
        raise ValueError("Sequence must be at least 6 amino acids long.")
    
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    invalid_residues = [residue for residue in seq.upper() if residue not in valid_amino_acids]
    
    if invalid_residues:
        raise ValueError(f"Invalid amino acids found in sequence: {', '.join(invalid_residues)}")

def get_dG_Int(seq: str) -> np.ndarray:
    """
    Get the intrinsic free energy contributions for a sequence. 
    The first and last residues are considered to be the caps and 
    do not contribute to the intrinsic free energy.

    Args:
        seq (str): The protein sequence.

    Returns:
        np.ndarray: The intrinsic free energy contributions for each amino acid in the sequence.
    """
    is_valid_amino_acid_sequence(seq)

    # Get the intrinsic energies for the sequence, but only for the residues in the helical conformation
    energy: np.ndarray = np.array([0.0 if i in [0, len(seq)-1] else table2.loc[AA, 'Intrinsic'] for i, AA in enumerate(seq)])

    # Custom case when Pro at N+1
    if seq[1] == 'P':
        energy[1] = 0.66
    return energy


def get_dG_Hbond(seq: str) -> float:
    """
    Get the free energy contribution for hydrogen bonding for a sequence.

    Args:
        seq (str): The protein sequence.

    Returns:
        float: The total free energy contribution for hydrogen bonding in the sequence.
    """
    is_valid_amino_acid_sequence(seq)

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
    is_valid_amino_acid_sequence(seq)

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
    is_valid_amino_acid_sequence(seq)   

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
    is_valid_amino_acid_sequence(seq)

    energy = np.zeros(len(seq))
    for i in range(len(seq) - 4):
        AAi = seq[i]
        AAi4 = seq[i + 4]
        energy[i] = table4b.loc[AAi, AAi4]
    return energy


def get_dG_Ncap(seq: str) -> np.ndarray:
    """
    Get the free energy contribution for N-terminal capping.

    Args:
        seq (str): The amino acid sequence.

    Returns:
        np.ndarray: The free energy contribution.
    """
    is_valid_amino_acid_sequence(seq)
    
    # Get the Ncap residue
    Ncap = seq[0]

    # initialize dG_Ncap
    dG_Ncap = table2.loc[Ncap, 'N_cap']

    # possible formation for capping box
    # when Glu (E) at N+3 (Harper & Rose, 1993;
    # Dasgupta & Bell, 1993)
    if seq[3] in ['E']:
        # AAAEAA should take us here but no farther
        dG_Ncap = table2.loc[Ncap, 'Capp_box']
        print('case 1')

        # special capping box case
        # when there is a Pro (P) at N+1
        if seq[1] == 'P':
            # APAEAA should take us here
            dG_Ncap += table2.loc[Ncap, 'Pro_N1']
            print('case 2')

    # Other capping box options:
    # Gln (Q) or Asp (D)
    elif seq[3] in ['Q', 'D']:
        # AAADAA and AAAQAA should take us here but no farther
        dG_Ncap = table2.loc[Ncap, 'Capp_box']
        print('case 3')

        # multiply negative values by 0.625   ----- But why? I thought this was supposed to provide stability, not reduce it.
        if dG_Ncap < 0.0:
            dG_Ncap *= 0.625
            print('case 4')

        # special capping box case
        # when there is a Pro (P) at N+1
        if seq[1] == 'P' and dG_Ncap < 0.0:
            # APADAA and APAQAA should take us here
            # multiply negative values by 0.625   ----- But why? I thought this was supposed to provide stability, not reduce it.
            dG_Ncap *= 0.625
            print('case 5')

    # Asp (D) + 2 special hydrogen bond, can be used to stabilize N-cap region
    # (Bell et al., 1992; Dasgurpta & Bell, 1993)
    if seq[2] == 'D' and dG_Ncap > 0.0:  # should I instead take the lowest value?
        dG_Ncap = table2.loc[Ncap, 'Asp_2']
        # only update if there is "no favorable" Ncap residue ----- What does this mean?
        # (Pro N+1 is sometimes more stabilizing)
        print('case 6')

    energy = np.zeros(len(seq))
    energy[0] = dG_Ncap

    return energy


def get_dG_Ccap(seq: str) -> np.ndarray:
    """
    Get the free energy contribution for C-terminal capping.

    Args:
        seq (str): The protein sequence.

    Returns:
        np.ndarray: The free energy contribution.
    """
    is_valid_amino_acid_sequence(seq)

    get_dG_Ccap = table2.loc[seq[-1], 'C_cap'] # Get the last amino acid in the sequence
    energy = np.zeros(len(seq))
    energy[-1] = get_dG_Ccap

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
    is_valid_amino_acid_sequence(seq)
    
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


def calc_K(dG_Hel: float, T: float=4.0) -> float:
    """
    Calculate the equilibrium constant K.

    Args:
        dG_Hel (float): The Helix free energy.
        T (float): Temperature in Celsius. Default is 4.0.

    Returns:
        float: The equilibrium constant K.
    """
    R = 1.987204258e-3 # kcal/mol/K
    celsius_2_kelvin = lambda T: T + 273.15
    RT = R * celsius_2_kelvin(T)
    return np.exp(-dG_Hel / RT)


class AGADIR(object):
    """
    AGADIR class for predicting helical propensity using AGADIR method.
    """

    def __init__(self, method: str = '1s', T: float = 4.0):
        """
        Initialize AGADIR object.

        Args:
            method (str): Method for calculating helical propensity. Must be one of ['r','1s'].
                'r' : Residue partition function.
                '1s': One-sequence approximation.
            T (float): Temperature in Celsius. Default is 4.0.
        """
        self.method_options = ['r', '1s']
        if method not in self.method_options:
            raise ValueError(
                "Method provided must be one of ['r','1s']; \
                'r' : Residue partition function. \
                '1s': One-sequence approximation. \
            See documentation and AGADIR papers for more information. \
            "
            )
        self._method = method
        self.T = T

        self.has_acetyl = False
        self.has_succinyl = False
        self.has_amide = False

        self.min_helix_length = 6

    def _calc_dG_Hel(self, seq: str) -> Tuple[np.float64, Dict[str, float]]:
        """
        Calculate the Helix free energy and its components.

        Args:
            seq (str): The helical segment sequence.

        Returns:
            Tuple[np.float64, Dict[str, float]]: The Helix free energy and its components.
        """
        j = len(seq)
        if j < self.min_helix_length:
            raise ValueError(f"Helix length must be at least {self.min_helix_length} amino acids long.")

        # intrinsic energies for the helical segment, excluding N- and C-terminal capping residues
        dG_Int = get_dG_Int(seq)

        # calculate dG_Hbond for the helical segment here
        dG_Hbond = get_dG_Hbond(seq)

        # side-chain interactions, excluding N- and C-terminal capping residues
        dG_i1_tot = get_dG_i1(seq)
        dG_i3_tot = get_dG_i3(seq)
        dG_i4_tot = get_dG_i4(seq)
        dG_SD = dG_i1_tot + dG_i3_tot + dG_i4_tot

        # capping energies, only for the first and last residues of the helix
        dG_Ncap = get_dG_Ncap(seq)
        dG_Ccap = get_dG_Ccap(seq)

        # non-hydrogen bond interactions
        dG_nonH = dG_Ncap + dG_Ccap

        # dipole interactions, excluding N- and C-terminal capping residues
        # the nomenclature is that of Richardson & Richardson (1988).
        dG_N_dipole, dG_C_dipole = get_dG_dipole(seq)
        dG_dipole = dG_N_dipole + dG_C_dipole

        # sum all components
        dG_Hel = sum(dG_Int) + dG_Hbond + sum(dG_SD) + sum(dG_nonH) + sum(dG_dipole)

        dG_dict = {
            'dG_Helix': dG_Hel,
            'dG_Int': dG_Int,
            'dG_Hbond': dG_Hbond,
            'dG_SD': dG_SD,
            'dG_nonH': dG_nonH,
            'dG_dipole': dG_dipole,
            'dG_N_dipole': dG_N_dipole,
            'dG_C_dipole': dG_C_dipole,
            'dG_i1_tot': dG_i1_tot,
            'dG_i3_tot': dG_i3_tot,
            'dG_i4_tot': dG_i4_tot,
            'dG_Ncap': dG_Ncap,
            'dG_Ccap': dG_Ccap
        }

        return dG_Hel, dG_dict

    def _calc_partition_fxn(self) -> None:
        """
        Calculate partition function for helical segments 
        by summing over all possible helices.
        """
        for i in range(0, len(self.result.seq) - self.min_helix_length + 1):  # for each position i
            for j in range(self.min_helix_length, len(self.result.seq) - i + 1):  # for each helix length j

                # get the relevant protein segment
                seq_segment = self.result.seq[i:i + j]

                # calculate dG_Hel and dG_dict
                dG_Hel, dG_dict = self._calc_dG_Hel(seq=seq_segment)

                # Add acetylation and amidation effects.
                # These are only considered for the first and last residues of the helix, 
                # and only if the peptide has been created in a way that they are present.
                if i == 0 and self.has_acetyl is True:
                    dG_Hel += -1.275
                    if self.result.seq[0] == 'A':
                        dG_Hel += -0.1

                elif i == 0 and self.has_succinyl is True:
                    dG_Hel += -1.775
                    if self.result.seq[0] == 'A':
                        dG_Hel += -0.1

                if (i + j == len(self.result.seq)) and (self.has_amide is True):
                    dG_Hel += -0.81
                    if self.result.seq[-1] == 'A':
                        dG_Hel += -0.1

                # calculate the partition function K
                K = calc_K(dG_Hel, self.T)
                self.result.K_tot_array[i + 1:i + j - 1] += K  # method='r', by definition helical region does not include caps
                self.result.K_tot += K  # method='1s'

        # if method='ms' (custom calculation here with result.dG_dict_mat)
        ### Not implemented yet ###

    def _calc_helical_propensity(self) -> None:
        """
        Calculate helical propensity based on the selected method.
        """
        # get per residue helical propensity
        if self._method == 'r':
            print('r')
            self.result.helical_propensity = 100 * self.result.K_tot_array / (1.0 + self.result.K_tot_array)

        elif self._method == '1s':
            print('1s')
            self.result.helical_propensity = 100 * self.result.K_tot_array / (1.0 + self.result.K_tot)

        # get overall percentage helix
        self.result.percent_helix = np.round(np.mean(self.result.helical_propensity), 2)

    def predict(self, seq: str) -> ModelResult:
        """
        Predict helical propensity for a given sequence.

        Args:
            seq (str): Input sequence.

        Returns:
            ModelResult: Object containing the predicted helical propensity.
        """
        seq = seq.upper()

        if len(seq) < self.min_helix_length:
            raise ValueError(f"Input sequence must be at least {self.min_helix_length} amino acids long.")

        # check for acylation and amidation
        if seq[0] == 'Z':
            self.has_acetyl = True
            seq = seq[1:]

        elif seq[0] == 'X':
            self.has_succinyl = True
            seq = seq[1:]

        if seq[-1] == 'B':
            self.has_amide = True
            seq = seq[:-1]

        if not set(list(seq)) <= set(list('ACDEFGHIKLMNPQRSTVWY')):
            raise ValueError('Parameter `seq` should contain only natural amino acids: ACDEFGHIKLMNPQRSTVWY.')

        self.result = ModelResult(seq)
        self._calc_partition_fxn()
        self._calc_helical_propensity()
        return self.result

if __name__ == "__main__":
    pass