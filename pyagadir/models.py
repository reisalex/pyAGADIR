
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
            seq (str): The sequence.
        """
        self.seq = seq
        self.int_array = np.zeros(len(seq))
        self.i1_array = np.zeros(len(seq))
        self.i3_array = np.zeros(len(seq))
        self.i4_array = np.zeros(len(seq))
        self.N_array = np.zeros(len(seq))
        self.C_array = np.zeros(len(seq))
        n = len(seq)
        self.dG_dict_mat = [None for j in range(5)] + [[None for _ in range(0, n - j)] for j in range(5, n)]
        self.K_tot = 0.0
        self.K_tot_array = np.zeros(len(seq))
        self.Z = 0.0
        self.Z_array = np.zeros(len(seq))
        self.helical_propensity = None
        self.percent_helix = None

    def __repr__(self) -> str:
        """
        Return a string representation of the helical propensity.

        Returns:
            str: The helical propensity.
        """
        return str(self.helical_propensity)

    def get_helical_propensity(self) -> np.ndarray:
        """
        Get the helical propensity.

        Returns:
            np.ndarray: The helical propensity.
        """
        return self.helical_propensity

    def get_percent_helix(self) -> float:
        """
        Get the percentage of helix.

        Returns:
            float: The percentage of helix.
        """
        return self.percent_helix


def get_dG_Int(AA: str) -> float:
    """
    Get the intrinsic free energy contribution.

    Args:
        AA (str): The amino acid.

    Returns:
        float: The intrinsic free energy contribution.
    """
    return table2.loc[AA, 'Intrinsic']


def get_dG_Hbond(j: int) -> float:
    """
    Get the free energy contribution for hydrogen bonding.

    Args:
        j (int): The index.

    Returns:
        float: The free energy contribution.
    """
    # A helix of 6 has two caps and four residues in the helix. 
    # The first 4 are considered to have zero net enthalpy and caps don't count.
    dG_Hbond = -0.775
    return dG_Hbond * max((j - 6), 0) 


def get_dG_i1(AAi: str, AAi1: str) -> float:
    """
    Get the free energy contribution for interaction between AAi and AAi1.

    Args:
        AAi (str): The first amino acid.
        AAi1 (str): The second amino acid.

    Returns:
        float: The free energy contribution.
    """
    charge = 1
    for AA in [AAi, AAi1]:
        if AA in set(['R', 'H', 'K']):
            charge *= 1
        elif AA in set(['D', 'E']):
            charge *= -1
        else:
            return 0.0
    else:
        return 0.05 * charge


def get_dG_i3(AAi: str, AAi3: str) -> float:
    """
    Get the free energy contribution for interaction between AAi and AAi3.

    Args:
        AAi (str): The first amino acid.
        AAi3 (str): The third amino acid.

    Returns:
        float: The free energy contribution.
    """
    return table4a.loc[AAi, AAi3]


def get_dG_i4(AAi: str, AAi4: str) -> float:
    """
    Get the free energy contribution for interaction between AAi and AAi4.

    Args:
        AAi (str): The first amino acid.
        AAi4 (str): The fourth amino acid.

    Returns:
        float: The free energy contribution.
    """
    return table4b.loc[AAi, AAi4]


def get_dG_Ncap(seq: str) -> float:
    """
    Get the free energy contribution for N-terminal capping.

    Args:
        seq (str): The amino acid sequence.

    Returns:
        float: The free energy contribution.
    """
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

    return dG_Ncap


def get_dG_Ccap(AA: str) -> float:
    """
    Get the free energy contribution for C-terminal capping.

    Args:
        AA (str): The amino acid.

    Returns:
        float: The free energy contribution.
    """
    return table2.loc[AA, 'C_cap']


def get_dG_dipole(seq: str) -> float:
    """
    Calculate the dipole free energy contribution.
    The nomenclature is that of Richardson & Richardson (1988),
    which is different from the one used in the AGADIR paper.
    Richardson considers the first and last helical residues as the caps.

    Args:
        seq (str): The amino acid sequence.

    Returns:
        float: The dipole free energy contribution.
    """
    N = len(seq)
    dG_N_dipole = 0.0
    dG_C_dipole = 0.0

    # N-term dipole contributions
    for i in range(0, min(N, 10)):
        AA = seq[i]
        dG_N_dipole += table3a.loc[AA, i]

    # C-term dipole contributions
    seq_inv = seq[::-1]
    for i in range(0, min(N, 10)):
        AA = seq_inv[i]
        dG_C_dipole += table3b.loc[AA, i*-1]

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
    celsius_2_kelvin = lambda T: T + 273.0
    RT = R * celsius_2_kelvin(T)
    return np.exp(-dG_Hel / RT)


class AGADIR(object):
    """
    AGADIR class for predicting helical propensity using AGADIR method.
    """

    method_options = ['r','1s']
    lambdas = {
        'r':  lambda result: result.K_tot_array / result.Z_array,
        '1s': lambda result: result.K_tot_array / result.Z
    }

    def __init__(self, method: str = '1s', T: float=4.0):
        """
        Initialize AGADIR object.

        Args:
            method (str): Method for calculating helical propensity. Must be one of ['r','1s'].
                'r' : Residue partition function.
                '1s': One-sequence approximation.
            T (float): Temperature in Celsius. Default is 4.0.
        """
        if method not in self.method_options:
            raise ValueError("Method provided must be one of ['r','1s']; \
                'r' : Residue partition function. \
                '1s': One-sequence approximation. \
            See documentation and AGADIR papers for more information. \
            ")
        self._method = method
        self._probability_fxn = self.lambdas[method]
        self.T = T

        self.has_acetyl = False
        self.has_succinyl = False
        self.has_amide = False

    def _initialize_params(self):
        """
        Initialize parameters for energy calculations.
        """
        seq = self.result.seq

        # get intrinsic energies
        for i in range(0, len(seq)):
            self.result.int_array[i] = get_dG_Int(seq[i])

        # get i,i+1 energies (coulombic)
        for i in range(0, len(seq)-1):
            self.result.i1_array[i] = get_dG_i1(seq[i], seq[i+1])

        # get i,i+3 energies (dG_SD)
        for i in range(0, len(seq)-3):
            self.result.i3_array[i] = get_dG_i3(seq[i], seq[i+3])

        # get i,i+4 energies (dG_SD)
        for i in range(0, len(seq)-4):
            self.result.i4_array[i] = get_dG_i4(seq[i], seq[i+4])
        
        # get Ncap energies
        for i in range(0, len(seq)-5):
            self.result.N_array[i] = get_dG_Ncap(seq[i:i+6])

        # get Ccap energies
        for i in range(5, len(seq)):
            self.result.C_array[i] = get_dG_Ccap(seq[i])

    def _calc_dG_Hel(self, i: int, j: int) -> Tuple[float, Dict[str, float]]:
        """
        Calculate the Helix free energy and its components.

        Args:
            i (int): The start index of the helical segment.
            j (int): The length of the helical segment.

        Returns:
            Tuple[float, Dict[str, float]]: The Helix free energy and its components.
        """

        # sum the intrinsic energies for the helical segment
        dG_Int = sum(self.result.int_array[i+1:i + j - 1]) # don't include caps

        # # custom case when Pro at N+1
        # if self.result.seq[i + 1] == 'P':
        #     dG_Int += (0.66 - 3.33)

        # have to calculate dG_Hbond here because it is only relevant for helices
        dG_Hbond = get_dG_Hbond(j)
        
        # sum side-chain interactions
        dG_i1_tot = sum(self.result.i1_array[i+1:i + j - 1 - 1]) # don't include caps
        dG_i3_tot = sum(self.result.i3_array[i+1:i + j - 1 - 3]) # don't include caps
        dG_i4_tot = sum(self.result.i4_array[i+1:i + j - 1 - 4]) # don't include caps
        dG_SD = dG_i1_tot + dG_i3_tot + dG_i4_tot

        # get capping energies
        dG_Ncap = self.result.N_array[i] # use only caps
        dG_Ccap = self.result.C_array[i + j - 1] # use only caps

        # add acetylation and amidation, if present
        if i == 0 and self.has_acetyl is True:
            dG_Ncap += -1.275
            if self.result.seq[0] == 'A':
                dG_Ncap += -0.1

        elif i == 0 and self.has_succinyl is True:
            dG_Ncap += -1.775
            if self.result.seq[0] == 'A':
                dG_Ncap += -0.1

        if i + j == len(self.result.seq) and self.has_amide is True:
            dG_Ccap += -0.81
            if self.result.seq[-1] == 'A':
                dG_Ccap += -0.1

        # sum non-hydrogen bond interactions
        dG_nonH = dG_Ncap + dG_Ccap

        # sum dipole interactions
        dG_N_dipole, dG_C_dipole = get_dG_dipole(self.result.seq[i+1:i + j - 1]) # caps are NOT needed, the nomenclature is that of Richardson & Richardson (1988).
        dG_dipole = dG_N_dipole + dG_C_dipole

        # sum all components
        dG_Hel = dG_Int + dG_Hbond + dG_SD + dG_nonH + dG_dipole

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

    def _calc_partition_fxn(self):
        """
        Calculate partition function for helical segments.
        """
        seq = self.result.seq

        n = len(seq)
        for j in range(6, n+1): # helix lengths (including caps)
            for i in range(0, n-j+1): # helical segment positions
                # print(i, j, i+j)

                # calculate dG_Hel and dG_dict
                dG_Hel, dG_dict = self._calc_dG_Hel(i, j)
                self.result.dG_dict_mat[j-1][i] = dG_dict

                # calculate K
                K = calc_K(dG_Hel, self.T)

                self.result.K_tot_array[i+1:i+j-1] += K # method='r', by definition helical region does not include caps
                self.result.K_tot += K # method='1s'

        # if method='ms' (custom calculation here with result.dG_dict_mat)

    def _calc_helical_propensity(self):
        """
        Calculate helical propensity based on the selected method.
        """
        self.result.Z_array = 1.0 + self.result.K_tot_array
        self.result.Z = 1.0 + self.result.K_tot

        self.result.helical_propensity = self._probability_fxn(self.result)
        self.result.percent_helix = np.round(np.mean(self.result.helical_propensity), 3)

    def predict(self, seq: str) -> ModelResult:
        """
        Predict helical propensity for a given sequence.

        Args:
            seq (str): Input sequence.

        Returns:
            ModelResult: Object containing the predicted helical propensity.
        """
        if not isinstance(seq, str):
            raise ValueError("Input sequence must be a string.")
        
        seq = seq.upper()
        
        if len(seq) < 6:
            raise ValueError("Input sequence must be at least 6 amino acids long.")
        
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
        self._initialize_params()
        self._calc_partition_fxn()
        self._calc_helical_propensity()
        return self.result

if __name__ == "__main__":
    pass