
import numpy as np
from typing import List, Tuple, Dict
from pyagadir import energies
from pyagadir.utils import is_valid_peptide_sequence, is_valid_index



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



class AGADIR(object):
    """
    AGADIR class for predicting helical propensity using AGADIR method.
    """

    def __init__(self, method: str = '1s', T: float = 4.0, M: float = 0.15, pH: float = 7.0):
        """
        Initialize AGADIR object.

        Args:
            method (str): Method for calculating helical propensity. Must be one of ['r','1s'].
                'r' : Residue partition function.
                '1s': One-sequence approximation.
            T (float): Temperature in Celsius. Default is 4.0.
            M (float): Ionic strength in Molar. Default is 0.15.
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
        self.T = T + 273.15
        self.molarity = M
        self.pH = pH
 
        self.has_acetyl = False
        self.has_succinyl = False
        self.has_amide = False

        self.min_helix_length = 6

    def _calc_dG_Hel(self, seq: str, i:int, j:int) -> Tuple[np.float64, Dict[str, float]]:
        """
        Calculate the Helix free energy and its components.

        Args:
            seq (str): The helical segment sequence.
            i (int): The starting position of the helical segment.
            j (int): The length of the helical segment.

        Returns:
            Tuple[np.float64, Dict[str, float]]: The Helix free energy and its components.
        """
        # intrinsic energies for the helical segment, excluding N- and C-terminal capping residues
        dG_Int = energies.get_dG_Int(seq, i, j)

        # "non-hydrogen bond" capping energies, only for the first and last residues of the helix
        dG_Ncap = energies.get_dG_Ncap(seq, i, j)
        dG_Ccap = energies.get_dG_Ccap(seq, i, j)
        dG_nonH = dG_Ncap + dG_Ccap
        # TODO dG_nonH might need further adjustment, see page 175 in lacroix paper

        # get hydrophobic staple motif energies
        dG_staple = energies.get_dG_staple(seq, i, j)

        # get schellman motif energies
        dG_schellman = energies.get_dG_schellman(seq, i, j)

        # calculate dG_Hbond for the helical segment here
        dG_Hbond = energies.get_dG_Hbond(seq, i, j)

        # side-chain interactions, excluding N- and C-terminal capping residues
        dG_i1_tot = energies.get_dG_i1(seq, i, j)
        dG_i3_tot = energies.get_dG_i3(seq, i, j)
        dG_i4_tot = energies.get_dG_i4(seq, i, j)
        dG_SD = dG_i1_tot + dG_i3_tot + dG_i4_tot

        # TODO: figure out how the dipole is supposed to be calculated
        # # dipole interactions, excluding N- and C-terminal capping residues
        # # the nomenclature is that of Richardson & Richardson (1988).
        # dG_N_dipole, dG_C_dipole = energies.get_dG_dipole(seq, i, j)
        # dG_dipole = dG_N_dipole + dG_C_dipole

        # get electrostatic interactions between N- and C-terminal capping charges and the helix macrodipole
        dG_N_term, dG_C_term = energies.get_dG_terminals(seq, i, j, self.molarity, self.pH, self.T)

        # get electrostatic energies between pairs of charged side chains
        dG_electrost = energies.get_dG_electrost(seq, i, j, self.molarity, self.pH, self.T)

        # modify by ionic strength according to equation 12 of the paper
        alpha = 0.15
        beta = 6.0
        dG_ionic = -alpha * (1 - np.exp(-beta * self.molarity))

        # make fancy printout for debugging and development
        for seq_idx, arr_idx in zip(range(i, i+j), range(j)):
            print(f'Helix: start= {i+1} end= {i+j}  length=  {j}')
            print(f'residue index = {seq_idx+1}')
            print(f'residue = {seq[seq_idx]}')
            print(f'g N term = {dG_N_term[arr_idx]:.4f}')
            print(f'g C term = {dG_C_term[arr_idx]:.4f}')
            print(f'g capping =   {dG_nonH[arr_idx]:.4f}')
            print(f'g intrinsic = {dG_Int[arr_idx]:.4f}')
            print(f'g dipole = ')
            print(f'gresidue = ')
            print('****************')
        print('Additional terms for helical segment')
        print(f'i,i+3 and i,i+4 side chain-side chain interaction = {sum(dG_SD):.4f}')
        print(f'g staple = {dG_staple:.4f}')
        print(f'g schellman = {dG_schellman:.4f}')
        print(f'dG_electrost = {dG_electrost:.4f}')
        print(f'main chain-main chain H-bonds = {dG_Hbond:.4f}')
        print(f'ionic strngth corr. from eq. 12 {dG_ionic:.4f}')

        # sum all components
        dG_Hel = sum(dG_Int) + sum(dG_nonH) +  sum(dG_SD) + dG_staple + dG_schellman + dG_Hbond + dG_ionic + sum(dG_N_term) + sum(dG_C_term) + dG_electrost # + sum(dG_dipole) 

        print(f'total Helix free energy = {dG_Hel:.4f}')
        print('==============================================')

        # TODO: do we need to return all these components? It was initally intended for the "ms" partition function calculation

        # dG_dict = {
        #     'dG_Helix': dG_Hel,
        #     'dG_Int': dG_Int,
        #     'dG_Hbond': dG_Hbond,
        #     'dG_SD': dG_SD,
        #     'dG_nonH': dG_nonH,
        #     'dG_dipole': dG_dipole,
        #     'dG_N_dipole': dG_N_dipole,
        #     'dG_C_dipole': dG_C_dipole,
        #     'dG_i1_tot': dG_i1_tot,
        #     'dG_i3_tot': dG_i3_tot,
        #     'dG_i4_tot': dG_i4_tot,
        #     'dG_Ncap': dG_Ncap,
        #     'dG_Ccap': dG_Ccap
        # }

        return dG_Hel, {}

    def _calc_K(self, dG_Hel: float) -> float:
        """
        Calculate the equilibrium constant K.

        Args:
            dG_Hel (float): The Helix free energy.

        Returns:
            float: The equilibrium constant K.
        """
        R = 1.987204258e-3 # kcal/mol/K
        return np.exp(-dG_Hel / (R * self.T))

    def _calc_partition_fxn(self) -> None:
        """
        Calculate partition function for helical segments 
        by summing over all possible helices.
        """
        # for i in range(0, len(self.result.seq) - self.min_helix_length + 1):  # for each position i
        #     for j in range(self.min_helix_length, len(self.result.seq) - i + 1):  # for each helix length j

        for j in range(self.min_helix_length, len(self.result.seq) + 1): # helix lengths (including caps)
            for i in range(0, len(self.result.seq) - j + 1): # helical segment positions

                # calculate dG_Hel and dG_dict
                dG_Hel, dG_dict = self._calc_dG_Hel(seq=self.result.seq, i=i, j=j)

                # TODO: these shuld be accounted for in the new table 1, verify this!
                # # Add acetylation and amidation effects.
                # # These are only considered for the first and last residues of the helix, 
                # # and only if the peptide has been created in a way that they are present.
                # if i == 0 and self.has_acetyl is True:
                #     dG_Hel += -1.275
                #     if self.result.seq[0] == 'A':
                #         dG_Hel += -0.1

                # elif i == 0 and self.has_succinyl is True:
                #     dG_Hel += -1.775
                #     if self.result.seq[0] == 'A':
                #         dG_Hel += -0.1

                # if (i + j == len(self.result.seq)) and (self.has_amide is True):
                #     dG_Hel += -0.81
                #     if self.result.seq[-1] == 'A':
                #         dG_Hel += -0.1

                # calculate the partition function K
                K = self._calc_K(dG_Hel)
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

        # ensure that the sequence is valid
        is_valid_peptide_sequence(seq)

        self.result = ModelResult(seq)
        self._calc_partition_fxn()
        self._calc_helical_propensity()
        return self.result

if __name__ == "__main__":
    pass
