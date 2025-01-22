from typing import Dict, Tuple

import numpy as np

from pyagadir.energies import EnergyCalculator
from pyagadir.utils import is_valid_peptide_sequence


class ModelResult:
    """
    Class containing the result of a model.
    """

    def __init__(self, seq: str) -> None:
        """
        Initialize the ModelResult object.

        Args:
            seq (str): The peptide sequence.
        """
        is_valid_peptide_sequence(seq)

        self.seq = seq
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

    def __init__(
        self, method: str = "1s", T: float = 4.0, M: float = 0.15, pH: float = 7.0
    ):
        """
        Initialize AGADIR object.

        Args:
            method (str): Method for calculating helical propensity. Must be one of ['r','1s'].
                'r' : Residue partition function.
                '1s': One-sequence approximation.
            T (float): Temperature in Celsius. Default is 4.0.
            M (float): Ionic strength in Molar (M, mol/L). Default is 0.15.
        """
        self.method_options = ["r", "1s"]
        if method not in self.method_options:
            raise ValueError(
                "Method provided must be one of ['r','1s']; \
                'r' : Residue partition function. \
                '1s': One-sequence approximation. \
            See documentation and AGADIR papers for more information. \
            "
            )

        # check for valid pH
        if not isinstance(pH, (float, int)):
            raise ValueError("pH must be a number")
        if pH < 0 or pH > 14:
            raise ValueError("pH must be between 0 and 14")

        # check for valid temperature
        if not isinstance(T, (float, int)):
            raise ValueError("Temperature must be a number")
        if T < -273.15:
            raise ValueError("Temperature must be above absolute zero")

        # check for valid molarity
        if not isinstance(M, (float, int)):
            raise ValueError("Molarity must be a number")
        if M < 0:
            raise ValueError("Molarity must be greater than 0")

        self._method = method
        self.T_celsius = T
        self.T_kelvin = T + 273.15
        self.molarity = M  # possibly need to adjust based on ion valency to actually get ionic strength here
        self.pH = pH

        self.n_cap = None
        self.c_cap = None
        self.has_acetyl = False
        self.has_succinyl = False
        self.has_amide = False

        self.min_helix_length = 6  # default from Lacroix 1998

    def _calc_dG_Hel(self, i: int, j: int) -> Tuple[np.float64, Dict[str, float]]:
        """
        Calculate free energy for a helical segment.

        Args:
            i (int): The starting position of the helical segment.
            j (int): The length of the helical segment.

        Returns:
            float: The Helix free energy.
        """
        # Intrinsic energies for the helical segment, excluding N- and C-terminal capping residues
        dG_Int = self.energy_calculator.get_dG_Int(i, j)

        # "non-hydrogen bond" capping energies, only for the first and last residues of the helix
        # TODO dG_nonH might need further adjustment, see page 175 in lacroix paper
        dG_Ncap = self.energy_calculator.get_dG_Ncap(i, j)
        dG_Ccap = self.energy_calculator.get_dG_Ccap(i, j)
        dG_nonH = dG_Ncap + dG_Ccap

        # get hydrophobic staple motif energies
        dG_staple = self.energy_calculator.get_dG_staple(i, j)

        # get schellman motif energies
        dG_schellman = self.energy_calculator.get_dG_schellman(i, j)

        # calculate hydrogen bond energies for the helical segment here
        dG_Hbond = self.energy_calculator.get_dG_Hbond(i, j)

        # side-chain interactions, excluding N- and C-terminal capping residues
        dG_i3_tot = self.energy_calculator.get_dG_i3(i, j)
        dG_i4_tot = self.energy_calculator.get_dG_i4(i, j)
        dG_SD = dG_i3_tot + dG_i4_tot

        # get the interactions between N- and C-terminal backbone charges and the helix macrodipole
        dG_N_term, dG_C_term = self.energy_calculator.get_dG_terminals_macrodipole(i, j)

        # get the interaction between charged side chains and the helix macrodipole
        dG_dipole_N, dG_dipole_C = self.energy_calculator.get_dG_sidechain_macrodipole(i, j)
        dG_dipole = dG_dipole_N + dG_dipole_C

        # get electrostatic energies between N- and C-terminal backbone charges and charged side chains
        # TODO 
        # dG_electrost_term_N, dG_electrost_term_C = self.energy_calculator.get_dG_terminals_sidechain_electrost(i, j)
        # dG_electrost_term = dG_electrost_term_N + dG_electrost_term_C

        # get electrostatic energies between pairs of charged side chains
        dG_electrost_sidechain = self.energy_calculator.get_dG_sidechain_sidechain_electrost(i, j)

        # modify by ionic strength according to equation 12 of the paper
        alpha = 0.15
        beta = 6.0
        dG_ionic = -alpha * (1 - np.exp(-beta * self.molarity))

        # sum all components
        dG_Hel = (
            sum(dG_Int)
            + sum(dG_nonH)
            + sum(dG_SD)
            + dG_staple
            + dG_schellman
            + dG_Hbond
            + dG_ionic
            + sum(dG_N_term)
            + sum(dG_C_term)
            + np.sum(dG_electrost_sidechain)
            + np.sum(dG_dipole)
        )

        # # make fancy printout for debugging and development
        # for seq_idx, arr_idx in zip(range(i, i + j), range(j)):
        #     print(f"Helix: start= {i+1} end= {i+j}  length=  {j}")
        #     print(f"residue index = {seq_idx+1}")
        #     print(f"residue = {self.result.seq[seq_idx]}")
        #     print(f"g N term = {dG_N_term[arr_idx]:.4f}")
        #     print(f"g C term = {dG_C_term[arr_idx]:.4f}")
        #     print(f"g capping =   {dG_nonH[arr_idx]:.4f}")
        #     print(f"g intrinsic = {dG_Int[arr_idx]:.4f}")
        #     print(f"g dipole N = {dG_dipole_N[arr_idx]:.4f}")
        #     print(f"g dipole C = {dG_dipole_C[arr_idx]:.4f}")
        #     print(f"g dipole total = {dG_dipole[arr_idx]:.4f}")
        #     print(f"gresidue = {dG_N_term[arr_idx] + dG_C_term[arr_idx] + dG_nonH[arr_idx] + dG_Int[arr_idx] + dG_dipole[arr_idx]:.4f}")
        #     print("****************")
        # print("Additional terms for helical segment")
        # print(f"i,i+3 and i,i+4 side chain-side chain interaction = {sum(dG_SD):.4f}")
        # print(f"g staple = {dG_staple:.4f}")
        # print(f"g schellman = {dG_schellman:.4f}")
        # print(f"dG_electrost = {np.sum(dG_electrost_sidechain):.4f}")
        # print(f"main chain-main chain H-bonds = {dG_Hbond:.4f}")
        # print(f"ionic strngth corr. from eq. 12 {dG_ionic:.4f}")
        # print(f"total Helix free energy = {dG_Hel:.4f}")
        # print("==============================================")

        return dG_Hel

    def _calc_K(self, dG_Hel: float) -> float:
        """
        Calculate the equilibrium constant K.

        Args:
            dG_Hel (float): The Helix free energy.

        Returns:
            float: The equilibrium constant K.
        """
        R = 1.987204258e-3  # kcal/mol/K
        return np.exp(-dG_Hel / (R * self.T_kelvin))

    def _calc_partition_fxn(self) -> None:
        """
        Calculate partition function for helical segments
        by summing over all possible helices.
        """
        # Special case for when there is a N-terminal modification (acetylation or succinylation)
        # The helix starting at the first residue can be one residue shorter because the modification acts as a capping residue
        # Get the energies for all helices starting at the first residue, allowing for a helix length of 1 residue shorter than the minimum length
        if self.has_acetyl or self.has_succinyl:
            for j in range(self.min_helix_length - 1, len(self.result.seq) + 1):
                i = 0
                dG_Hel = self._calc_dG_Hel(i=i, j=j)
                K = self._calc_K(dG_Hel)
                self.result.K_tot_array[i : i + j - 1] += K
                self.result.K_tot += K

        # Special case for when there is a C-terminal modification (amidation)
        # The helix ending at the last residue can be one residue shorter because the modification acts as a capping residue
        # Get the energies for all helices ending at the last residue, allowing for a helix length of 1 residue shorter than the minimum length
        if self.has_amide:
            for j in range(self.min_helix_length - 1, len(self.result.seq) + 1):
                i = len(self.result.seq) - j
                dG_Hel = self._calc_dG_Hel(i=i, j=j)
                K = self._calc_K(dG_Hel)
                self.result.K_tot_array[i + 1 : i + j] += K
                self.result.K_tot += K

        # General case for all other helices
        # Here the first and last residues act as capping residues, these must be the minimum length of 6 residues
        for j in range(
            self.min_helix_length, len(self.result.seq) + 1
        ):  # helix lengths (including capping residues)
            for i in range(
                0, len(self.result.seq) - j + 1
            ):  # helical segment positions

                # calculate dG_Hel and dG_dict
                dG_Hel = self._calc_dG_Hel(i=i, j=j)

                # calculate the partition function K
                K = self._calc_K(dG_Hel)

                # add the partition function to the total partition function
                self.result.K_tot_array[
                    i + 1 : i + j - 1
                ] += K  # method='r', by definition helical region does not include capping residues, hence i+1 and i+j-1
                self.result.K_tot += K  # method='1s'

    def _calc_helical_propensity(self) -> None:
        """
        Calculate helical propensity based on the selected method.
        """
        # get per residue helical propensity
        if self._method == "r":
            self.result.helical_propensity = (
                100 * self.result.K_tot_array / (1.0 + self.result.K_tot_array)
            )

        elif self._method == "1s":
            self.result.helical_propensity = (
                100 * self.result.K_tot_array / (1.0 + self.result.K_tot)
            )

        # get overall percentage helix
        self.result.percent_helix = np.round(np.mean(self.result.helical_propensity), 2)

    def predict(self, seq: str, ncap: str = None, ccap: str = None) -> ModelResult:
        """
        Predict helical propensity for a given sequence.

        Args:
            seq (str): Input sequence.
            ncap (str): N-terminal capping modification (acetylation='Z', succinylation='X').
            ccap (str): C-terminal capping modification (amidation='B').

        Returns:
            ModelResult: Object containing the predicted helical propensity.
        """
        # check for valid sequence
        is_valid_peptide_sequence(seq)
        seq = seq.upper()

        # check for valid ncap
        if ncap is not None:
            if ncap not in ["Z", "X"]:
                raise ValueError(
                    f"Invalid N-terminal capping modification: {ncap}, must be None,'Z' or 'X'"
                )
            elif ncap == "Z":
                self.has_acetyl = True
                self.n_cap = "Z"
            elif ncap == "X":
                self.has_succinyl = True
                self.n_cap = "X"

        # check for valid ccap
        if ccap is not None:
            if ccap not in ["B"]:
                raise ValueError(
                    f"Invalid C-terminal capping modification: {ccap}, must be None or 'B'"
                )
            elif ccap == "B":
                self.has_amide = True
                self.c_cap = "B"

        # check for valid sequence length
        if len(seq) < self.min_helix_length:
            raise ValueError(
                f"Input sequence must be at least {self.min_helix_length} amino acids long."
            )
        
        print(f"Predicting helical propensity for sequence: {seq}, method: {self._method}, T(C): {self.T_celsius}, M: {self.molarity}, pH: {self.pH}, ncap: {self.n_cap}, ccap: {self.c_cap}")

        # create energy calculator instance
        self.energy_calculator = EnergyCalculator(
            seq,
            pH=self.pH,
            T=self.T_celsius,
            ionic_strength=self.molarity,
            min_helix_length=self.min_helix_length,
            has_acetyl=self.has_acetyl,
            has_succinyl=self.has_succinyl,
            has_amide=self.has_amide,
        )

        # initialize the result object
        self.result = ModelResult(seq)

        # calculate the partition function and helical propensity
        self._calc_partition_fxn()
        self._calc_helical_propensity()

        return self.result


if __name__ == "__main__":
    pass
