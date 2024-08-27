
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
        self.dG_dict_mat = [None for j in range(5)] + [[None for _ in range(0, n - j)] for j in range(5, n)] # helix length is at least 6 but we zero-index
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
    
    def get_sequence(self) -> str:
        """
        Get the sequence.

        Returns:
            str: The sequence.
        """
        return self.seq

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
    
    # def get_initialized_params(self) -> str:
    #     """
    #     Output the initialized parameters for energy calculations.
    #     """
    #     header_just = 21
    #     table_just = 11
    #     output = []

    #     output.append('Thanks for using AGADIR')
    #     output.append('')
    #     output.append('These are the initialized parameters for energy calculations')
    #     output.append('')
    #     output.append(f'{"pH".ljust(header_just)}7.0')
    #     output.append(f'{"Temperature (K)".ljust(header_just)}278')
    #     output.append(f'{"Ionic strength (M)".ljust(header_just)}0.1')
    #     output.append('')
    #     output.append(f'{"Nterm".ljust(header_just)}free')
    #     output.append(f'{"Cterm".ljust(header_just)}free')
    #     output.append('')
    #     output.append(f'{"Peptide".ljust(header_just)}{self.seq}')
    #     output.append('')
    #     output.append('dG values in kcal/mol')
    #     output.append('')

    #     # intrinsic energies
    #     headers = ["res", "aa", "Intrinsic", "i,i+1", "i,i+3", "i,i+4", "Ncap", "Ccap"]
    #     formatted_headers = ''.join([h.ljust(table_just) for h in headers])
    #     output.append(formatted_headers)

    #     for i, AA in enumerate(self.seq):
    #         values = [i+1,
    #                   AA, 
    #                   round(self.int_array[i], 2), 
    #                   round(self.i1_array[i], 2), 
    #                   round(self.i3_array[i], 2), 
    #                   round(self.i4_array[i], 2), 
    #                   round(self.N_array[i], 2), 
    #                   round(self.C_array[i], 2)]
    #         formatted_values = ''.join([str(v).ljust(table_just) for v in values])
    #         output.append(formatted_values)

    #     output.append('')
    #     output.append('Legend:')
    #     output.append('Intrinsic = intrinsic helix propensity')
    #     output.append('i,i+1 = interaction between i and i+1')
    #     output.append('i,i+3 = interaction between i and i+3')
    #     output.append('i,i+4 = interaction between i and i+4')
    #     output.append('Ncap = N-terminal capping')
    #     output.append('Ccap = C-terminal capping')

    #     return '\n'.join(output)
    
    # def get_separate_helix_contributions(self) -> str:
    #     """
    #     Get the contributions of each amino acid to the helical propensity,
    #     separated by helix length.
    #     """
    #     header_just = 21
    #     table_just = 11
    #     output = []

    #     output.append('Thanks for using AGADIR')
    #     output.append('')
    #     output.append('These are the contributions of each amino acid to the helical propensity, separated by helix length')
    #     output.append('')
    #     output.append(f'{"pH".ljust(header_just)}7.0')
    #     output.append(f'{"Temperature (K)".ljust(header_just)}278')
    #     output.append(f'{"Ionic strength (M)".ljust(header_just)}0.1')
    #     output.append('')
    #     output.append(f'{"Nterm".ljust(header_just)}free')
    #     output.append(f'{"Cterm".ljust(header_just)}free')
    #     output.append('')
    #     output.append(f'{"Peptide".ljust(header_just)}{self.seq}')
    #     output.append('')
    #     output.append('dG values in kcal/mol')
    #     output.append('')

    #     # headers
    #     headers = ["res","aa"] + [f"H_len_{j+1}" for j in range(5, len(self.seq))]
    #     formatted_headers = ''.join([h.ljust(table_just) for h in headers])
    #     output.append(formatted_headers)
        
    #     # intrinsic energies
    #     for i, AA in enumerate(self.seq):
    #         values = [i+1, AA]
    #         for j, my_list in enumerate(self.dG_dict_mat):
    #             if my_list is None:
    #                 continue

    #             if len(my_list) <= i:
    #                 continue

    #             my_dict = my_list[i]
    #             if my_dict is None:
    #                 raise ValueError(f"Helix length {j+1} does not have a contribution for position {i+1}.")
                

    #             values.append(round(my_dict['dG_Int'], 2))
                           
    #         formatted_values = ''.join([str(v).ljust(table_just) for v in values])
    #         output.append(formatted_values)

    #     output.append('')
    #     return '\n'.join(output)

    
    # def get_total_helix_contributions(self) -> str:
    #     """
    #     Get the contributions of each amino acid to the helical propensity,
    #     including all of the different energy terms.
    #     """
    #     header_just = 21
    #     table_just = 11
    #     output = []

    #     # header with general information
    #     output.append('Thanks for using AGADIR')
    #     output.append('')
    #     output.append(f'{"pH".ljust(header_just)}7.0') # default pH
    #     output.append(f'{"Temperature (K)".ljust(header_just)}278') # default temperature
    #     output.append(f'{"Ionic strength (M)".ljust(header_just)}0.1') # default ionic strength
    #     output.append('')
    #     output.append(f'{"Nterm".ljust(header_just)}free')
    #     output.append(f'{"Cterm".ljust(header_just)}free')
    #     output.append('')
    #     output.append(f'{"Peptide".ljust(header_just)}{self.seq}')
    #     output.append('')
    #     output.append('dG values in kcal/mol')

    #     # table of the actual contributions
    #     headers = ["res", 
    #                "aa", 
    #                "Sum", 
    #                "Int", 
    #                "Hbond", 
    #                "N_dipole",
    #                "C_dipole",
    #                "all_dipole", 
    #                "i1_tot",
    #                "i3_tot",
    #                "i4_tot",
    #                "SD", 
    #                "Ncap",
    #                "Ccap",
    #                "all_cap"]
        
    #     formatted_headers = ''.join([h.ljust(table_just) for h in headers])
    #     output.append(formatted_headers)

    #     # for each position accumulate the contributions of all helices of different lengths
    #     dG_Helix = [0.0 for _ in range(len(self.seq))]
    #     dG_Int = [0.0 for _ in range(len(self.seq))]
    #     dG_Hbond = [0.0 for _ in range(len(self.seq))]
    #     dG_N_dipole = [0.0 for _ in range(len(self.seq))]
    #     dG_C_dipole = [0.0 for _ in range(len(self.seq))]
    #     dG_dipole = [0.0 for _ in range(len(self.seq))]
    #     dG_i1_tot = [0.0 for _ in range(len(self.seq))]
    #     dG_i3_tot = [0.0 for _ in range(len(self.seq))]
    #     dG_i4_tot = [0.0 for _ in range(len(self.seq))]
    #     dG_SD = [0.0 for _ in range(len(self.seq))]
    #     dG_Ncap = [0.0 for _ in range(len(self.seq))]
    #     dG_Ccap = [0.0 for _ in range(len(self.seq))]
    #     dG_nonH = [0.0 for _ in range(len(self.seq))]
    #     for j, my_list in enumerate(self.dG_dict_mat): # for each helix length
    #         if my_list is None:
    #             continue

    #         for i, my_dict in enumerate(my_list): # for each position in the sequence
                
    #             dG_Helix[i] += my_dict['dG_Helix']
    #             dG_Int[i] += my_dict['dG_Int']
    #             dG_Hbond[i] += my_dict['dG_Hbond']
    #             dG_N_dipole[i] += my_dict['dG_N_dipole']
    #             dG_C_dipole[i] += my_dict['dG_C_dipole']
    #             dG_dipole[i] += my_dict['dG_dipole']
    #             dG_i1_tot[i] += my_dict['dG_i1_tot']
    #             dG_i3_tot[i] += my_dict['dG_i3_tot']
    #             dG_i4_tot[i] += my_dict['dG_i4_tot']
    #             dG_SD[i] += my_dict['dG_SD']
    #             dG_Ncap[i] += my_dict['dG_Ncap']
    #             dG_Ccap[i] += my_dict['dG_Ccap']
    #             dG_nonH[i] += my_dict['dG_nonH']

    #     for i, aa in enumerate(self.seq):
    #         num_decimals = 2
    #         values = [i+1, 
    #                   aa,
    #                   round(dG_Helix[i], num_decimals), 
    #                     round(dG_Int[i], num_decimals),
    #                     round(dG_Hbond[i], num_decimals),
    #                     round(dG_N_dipole[i], num_decimals),
    #                     round(dG_C_dipole[i], num_decimals),
    #                     round(dG_dipole[i], num_decimals),
    #                     round(dG_i1_tot[i], num_decimals),
    #                     round(dG_i3_tot[i], num_decimals),
    #                     round(dG_i4_tot[i], num_decimals),
    #                     round(dG_SD[i], num_decimals),
    #                     round(dG_Ncap[i], num_decimals),
    #                     round(dG_Ccap[i], num_decimals),
    #                     round(dG_nonH[i], num_decimals),]
            
    #         formatted_values = ''.join([str(v).ljust(table_just) for v in values])
    #         output.append(formatted_values)

    #     output.append('')
    #     output.append(f'{"Percentage helix".ljust(header_just)}{self.percent_helix*100 :.2f}')
    #     output.append('')
    #     output.append('Legend:')
    #     output.append('Sum = Int + Hbond + SD + all_cap + all_dipole')
    #     output.append('all_dipole = N_dipole + C_dipole')
    #     output.append('SD = i1_tot + i3_tot + i4_tot')
    #     output.append('all_cap = Ncap + Ccap')


    #     return '\n'.join(output)


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

    # get the intrinsic energies for the sequence
    energy = np.array([0.0 if i in [0, len(seq)-1] else table2.loc[AA, 'Intrinsic'] for i, AA in enumerate(seq)])

    # custom case when Pro at N+1
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


def get_dG_Ncap(seq: str) -> float:
    """
    Get the free energy contribution for N-terminal capping.

    Args:
        seq (str): The amino acid sequence.

    Returns:
        float: The free energy contribution.
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

    return dG_Ncap


def get_dG_Ccap(seq: str) -> float:
    """
    Get the free energy contribution for C-terminal capping.

    Args:
        seq (str): The protein sequence.

    Returns:
        float: The free energy contribution.
    """
    is_valid_amino_acid_sequence(seq)

    Ccap = seq[-1]  # Get the last amino acid in the sequence
    return table2.loc[Ccap, 'C_cap']


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

        self.min_helix_length = 6

    def _calc_dG_Hel(self, seq: str) -> Tuple[float, Dict[str, float]]:
        """
        Calculate the Helix free energy and its components.

        Args:
            seq (str): The helical segment sequence.

        Returns:
            Tuple[float, Dict[str, float]]: The Helix free energy and its components.
        """
        j = len(seq)
        if j < self.min_helix_length:
            raise ValueError(f"Helix length must be at least {self.min_helix_length} amino acids long.")

        # get the intrinsic energies for the helical segment, excluding N- and C-terminal capping ridues
        dG_Int = get_dG_Int(seq)

        # calculate dG_Hbond for the helical segment here
        dG_Hbond = get_dG_Hbond(seq)
        
        # sum side-chain interactions, excluding N- and C-terminal capping ridues
        dG_i1_tot = get_dG_i1(seq)
        dG_i3_tot = get_dG_i3(seq)
        dG_i4_tot = get_dG_i4(seq)
        dG_SD = sum(dG_i1_tot) + sum(dG_i3_tot) + sum(dG_i4_tot)

        # get capping energies, only for the first and last residues of the helix
        dG_Ncap = get_dG_Ncap(seq)
        dG_Ccap = get_dG_Ccap(seq)

        # sum non-hydrogen bond interactions
        dG_nonH = dG_Ncap + dG_Ccap

        # sum dipole interactions, excluding N- and C-terminal capping ridues
        # the nomenclature is that of Richardson & Richardson (1988).
        dG_N_dipole, dG_C_dipole = get_dG_dipole(seq)
        dG_dipole = sum(dG_N_dipole) + sum(dG_C_dipole)

        # sum all components
        dG_Hel = sum(dG_Int) + dG_Hbond + dG_SD + dG_nonH + dG_dipole

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
        Calculate partition function for helical segments 
        by summing over all possible helices.
        """
        for j in range(self.min_helix_length, len(self.result.seq) + 1): # helix lengths (including caps)
            for i in range(0, len(self.result.seq) - j + 1): # helical segment positions
                # print('i:', i, 'j:', j, 'seq:', self.result.seq[i:i+j])

                # get the relevant protein segment
                seq_segment = self.result.seq[i:i + j]

                # calculate dG_Hel and dG_dict
                dG_Hel, dG_dict = self._calc_dG_Hel(seq=seq_segment)
                self.result.dG_dict_mat[j-1][i] = dG_dict

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

                # calculate K
                K = calc_K(dG_Hel, self.T)

                self.result.K_tot_array[i + 1:i + j - 1] += K # method='r', by definition helical region does not include caps
                self.result.K_tot += K # method='1s'

        # if method='ms' (custom calculation here with result.dG_dict_mat)
        ### Not implemented yet ###

    def _calc_helical_propensity(self):
        """
        Calculate helical propensity based on the selected method.
        """
        self.result.Z_array = 1.0 + self.result.K_tot_array
        self.result.Z = 1.0 + self.result.K_tot

        self.result.helical_propensity = 100 * self._probability_fxn(self.result)
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