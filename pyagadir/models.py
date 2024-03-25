
from importlib.resources import files
import numpy  as np
import pandas as pd
from typing import List
import numpy as np
import numpy as np

# get params
datapath = files('pyagadir.data')

table2  = pd.read_csv(
    datapath.joinpath('table2.csv'),
    index_col='AA',
).astype(float)

table3a = pd.read_csv(
    datapath.joinpath('table3a.csv'),
    index_col='AA',
).astype(float)
table3a.columns = table3a.columns.astype(int)

table3b = pd.read_csv(
    datapath.joinpath('table3b.csv'),
    index_col='AA',
).astype(float)
table3b.columns = table3b.columns.astype(int)

table4a = pd.read_csv(
    datapath.joinpath('table4a.csv'),
    index_col='index',
).astype(float)

table4b = pd.read_csv(
    datapath.joinpath('table4b.csv'),
    index_col='index',
).astype(float)

def get_dG_Int(AA):
    return table2.loc[AA,'Intrinsic']

dG_Hbond = -0.775
def get_dG_Hbond(j):
    return dG_Hbond*max((j-4),0)

def get_dG_i1(AAi,AAi1):
    charge = 1
    for AA in [AAi,AAi1]:
        if AA in set(['R','H','K']): # pos
            charge *= 1
        elif AA in set(['D','E']): # neg
            charge *= -1
        else:
            return 0.0
    else:
        return 0.05*charge

def get_dG_i3(AAi,AAi3):
    return table4a.loc[AAi, AAi3]

def get_dG_i4(AAi,AAi4):
    return table4b.loc[AAi, AAi4]

# seq :: first 4 AA considered for dG_Ncap
def get_dG_Ncap(seq):

    Ncap = seq[0]

    # possible formation for capping box
    # when Glu (E) at N+3 (Harper & Rose, 1993;
    # Dasgupta & Bell, 1993)
    if seq[3] in ['E','Q','D']:
        
        dG_Ncap = table2.loc[Ncap,'Capp_box']
    
        # Other capping box options:
        # Gln (Q) or Asp (D)
        if seq[3] in ['Q','D'] and dG_Ncap < 0.0:
            dG_Ncap *= 0.625

        # special capping box case
        # when there is a Pro (P) at N+1
        if seq[1] == 'P':
            dG_Ncap += table2.loc[Ncap,'Pro_N1']

    else:
        dG_Ncap = table2.loc[Ncap,'N_cap']

    # Asp (D) + 2 special hygroden bond
    # (Bell et al., 1992; Dasgurpta & Bell, 1993)
    if seq[2] == 'D':
        dG_Ncap = min(table2.loc[Ncap,'Asp_2'], dG_Ncap)
        # only update if more negative
        # (Pro N+1 is sometimes more stabilizing)

    return dG_Ncap

def get_dG_Ccap(AA):
    return table2.loc[AA,'C_cap']

def get_dG_dipole(seq):

    N = len(seq)
    dG_dipole = 0.0

    # N-term contribution (negative)
    for i in range(0,min(N,10)):
        AA = seq[i]
        dG_dipole += table3a.loc[AA,i]

    # C-term contributions (positive)
    for i in range(-1,-1*min(N,10)-1,-1):
        AA = seq[i]
        dG_dipole += table3b.loc[AA,i]

    return dG_dipole

def calc_dG_Hel(i, j, result):
    # where i is the start and j is the length of helical segment

    dG_Int    = sum(result.int_array[i:i+j])

    # custom case when Pro at N+1
    if result.seq[i+1] == 'P':
        dG_Int += (0.66 - 3.33)

    dG_Hbond  = get_dG_Hbond(j)
    dG_i1_tot = sum(result.i1_array[i:i+j-1])
 
    dG_i3_tot = sum(result.i3_array[i:i+j-3])
    dG_i4_tot = sum(result.i4_array[i:i+j-4])
    dG_SD     = dG_i1_tot + dG_i3_tot + dG_i4_tot

    dG_Ncap   = result.N_array[i]
    dG_Ccap   = result.C_array[i+j-1]
    dG_nonH   = dG_Ncap + dG_Ccap
    
    dG_dipole = get_dG_dipole(result.seq[i:i+j])

    dG_Hel = dG_Int + dG_Hbond + dG_SD + dG_nonH + dG_dipole

    dG_dict = {
        'dG_Helix': dG_Hel,
        'dG_Int': dG_Int,
        'dG_Hbond': dG_Hbond,
        'dG_SD': dG_SD,
        'dG_nonH': dG_nonH,
        'dG_dipole': dG_dipole,
        'dG_i1_tot': dG_i1_tot,
        'dG_i3_tot': dG_i3_tot,
        'dG_i4_tot': dG_i4_tot,
        'dG_Ncap': dG_Ncap,
        'dG_Ccap': dG_Ccap
    }

    return dG_Hel, dG_dict


def calc_K(dG_Hel: float, T: float = 5.0) -> float:
    """
    Calculate the equilibrium constant K.

    Args:
        dG_Hel (float): The Helix free energy.
        T (float, optional): Temperature in Celsius. Defaults to 5.0.

    Returns:
        float: The equilibrium constant K.
    """
    R = 1.987204258e-3 # kcal/mol/K
    celsius_2_kelvin = lambda T: T + 273.0
    RT = R * celsius_2_kelvin(T)
    return np.exp(-dG_Hel / RT)


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
        self.dG_dict_mat = [None for j in range(6)] + [[None for _ in range(0, n - j + 1)] for j in range(6, n + 1)]
        self.K_tot = 0.0
        self.K_tot_array = np.zeros(len(seq))
        self.Z = 0.0
        self.Z_array = np.zeros(len(seq))
        self.helical_propensity = np.zeros(len(seq))
        self.percent_helix = 0.0

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


class AGADIR(object):
    """
    AGADIR class for predicting helical propensity using AGADIR method.
    """

    method_options = ['r','1s']
    lambdas = {
        'r':  lambda result: result.K_tot_array / result.Z_array,
        '1s': lambda result: result.K_tot_array / result.Z
    }

    def __init__(self, method: str = '1s'):
        """
        Initialize AGADIR object.

        Args:
            method (str): Method for calculating helical propensity. Must be one of ['r','1s'].
                'r' : Residue partition function.
                '1s': One-sequence approximation.
        """
        if method not in self.method_options:
            raise ValueError("Method provided must be one of ['r','1s']; \
                'r' : Residue partition function. \
                '1s': One-sequence approximation. \
            See documentation and AGADIR papers for more information. \
            ")
        self._method = method
        self._probability_fxn = self.lambdas[method]

    def _calc_helical_propensity(self):
        """
        Calculate helical propensity based on the selected method.
        """
        result = self.result
        result.Z_array = 1.0 + result.K_tot_array
        result.Z       = 1.0 + result.K_tot
        result.helical_propensity = self._probability_fxn(result)
        result.percent_helix = np.round(np.mean(result.helical_propensity),3)

    def _initialize_params(self):
        """
        Initialize parameters for energy calculations.
        """
        result = self.result
        seq = result.seq

        # get intrinsic energies
        for i in range(0,len(seq)):
            result.int_array[i] = get_dG_Int(seq[i])

        # get i,i+1 energies (coulombic)
        for i in range(0,len(seq)-1):
            result.i1_array[i] = get_dG_i1(seq[i],seq[i+1])

        # get i,i+3 energies (dG_SD)
        for i in range(0,len(seq)-3):
            result.i3_array[i] = get_dG_i3(seq[i],seq[i+3])

        # get i,i+4 energies (dG_SD)
        for i in range(0,len(seq)-4):
            result.i4_array[i] = get_dG_i4(seq[i],seq[i+4])
        
        # get Ncap energies
        for i in range(0,len(seq)-5):
            result.N_array[i] = get_dG_Ncap(seq[i:i+6])

        # get Ccap energies
        for i in range(5,len(seq)):
            result.C_array[i] = get_dG_Ccap(seq[i])

    def _calc_partition_fxn(self):
        """
        Calculate partition function for helical segments.
        """
        result = self.result
        seq    = self.result.seq

        n = len(seq)
        for j in range(6,n+1): # helical segment lengths
            for i in range(0,n-j+1): # helical segment positions
                dG_Hel, dG_dict = calc_dG_Hel(i, j, result)
                result.dG_dict_mat[j][i] = dG_dict
                K = calc_K(dG_Hel)
                result.K_tot_array[i:i+j] += K # method='r'
                result.K_tot += K # method='1s'

        # if method='ms' (custom calculation here with result.dG_dict_mat)

    def predict(self, seq: str) -> ModelResult:
        """
        Predict helical propensity for a given sequence.

        Args:
            seq (str): Input sequence.

        Returns:
            ModelResult: Object containing the predicted helical propensity.
        """
        result = self.result = ModelResult(seq)
        self._initialize_params()
        self._calc_partition_fxn()
        self._calc_helical_propensity()
        return result

if __name__ == "__main__":
    pass