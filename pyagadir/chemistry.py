from typing import Dict
import warnings
import math


def calculate_ionic_strength(ion_concentrations: Dict[str, float]) -> float:
    """
    Calculate the ionic strength of a solution containing various ions commonly used in biological systems.

    Args:
        ion_concentrations: A dictionary with ion names as keys and their
                            concentrations (in M) as values.

    Returns:
        The ionic strength of the solution in M.
    """
    ion_charges = {
        # Monovalent cations
        "Na+": 1,
        "K+": 1,
        "Li+": 1,
        "NH4+": 1,
        # Divalent cations
        "Ca2+": 2,
        "Mg2+": 2,
        "Mn2+": 2,
        "Fe2+": 2,
        "Zn2+": 2,
        "Cu2+": 2,
        # Monovalent anions
        "Cl-": -1,
        "Br-": -1,
        "I-": -1,
        "F-": -1,
        "NO3-": -1,
        "HCO3-": -1,
        # Divalent anions
        "SO42-": -2,
        "HPO42-": -2,
        # Trivalent anions
        "PO43-": -3,
    }

    # Check for negative or zero concentrations
    if any(conc <= 0 for conc in ion_concentrations.values()):
        raise ValueError("Ion concentrations cannot be negative or zero")

    # Check for any unrecognized ions
    unrecognized = [ion for ion in ion_concentrations if ion not in ion_charges]
    if unrecognized:
        warnings.warn(
            f"The following ions were not recognized and were not included in the calculation: {', '.join(unrecognized)}"
        )

    inoic_strength = 0.5 * sum(
        conc * ion_charges[ion] ** 2
        for ion, conc in ion_concentrations.items()
        if ion in ion_charges
    )

    return inoic_strength


def adjust_pKa(T: float, pKa_ref: float, deltaG: float) -> float:
    """
    Adjust the pKa of a residue based on its free energy difference.
    Eq. 8-9 in Lacroix 1998

    Args:
        T (float): Temperature in Kelvin.
        pKa_ref (float): Reference pKa value.
        deltaG (float): Free energy difference (kcal/mol).

    Returns:
        float: Adjusted pKa value.
    """
    R = 1.987e-3  # kcal/(mol K)
    return pKa_ref + deltaG / (2.3 * R * T)


def acidic_residue_ionization(pH: float, pKa: float) -> float:
    """
    Calculate the degree of ionization for acidic residues.
    Uses the Henderson-Hasselbalch equation.
    Eq. 10 in Lacroix 1998.

    Args:
        pH (float): The pH of the solution.
        pKa (float): Adjusted pKa value.

    Returns:
        float: Ionization state (-1 for fully deprotonated, 0 for neutral).
    """
    return -1 / (1 + 10 ** (pKa - pH))


def basic_residue_ionization(pH: float, pKa: float) -> float:
    """
    Calculate the degree of ionization for basic residues.
    Uses the Henderson-Hasselbalch equation.
    Eq. 11 in Lacroix 1998.

    Args:
        pH (float): The pH of the solution.
        pKa (float): Adjusted pKa value.

    Returns:
        float: Ionization state (1 for fully protonated, 0 for neutral).
    """
    return 1 / (1 + 10 ** (pH - pKa))


def calculate_permittivity(T: float) -> float:
    """Calculate the relative permittivity of water at a given temperature.
    From J. Am. Chem. Soc. 1950, 72, 7, 2844-2847
    https://doi.org/10.1021/ja01163a006

    Args:
        T (float): Temperature in Kelvin.

    Returns:
        float: The relative permittivity of water.
    """
    epsilon_r = (
        5321 / T + 233.76 - 0.9297 * T + 0.1417 * 1e-2 * T**2 - 0.8292 * 1e-6 * T**3
    )
    return epsilon_r


def debye_huckel_screening_parameter(ionic_strength: float, T: float) -> float:
    """Calculate the Debye-Huckel screening parameter K.
    Equation 7 from Lacroix, 1998.

    Args:
        ionic_strength (float): Ionic strength of the solution in mol/L.
        T (float): Temperature in Kelvin.

    Returns:
        float: The Debye-Huckel screening parameter K.
    """
    # TODO: Is this function actually needed anywhere???

    N_A = 6.022e23  # Avogadro's number in mol^-1
    e = 1.602e-19  # Elementary charge in Coulombs
    k_B = 1.38e-23  # Boltzmann constant in J/K

    # Convert ionic strength to mol/m**3
    ionic_strength = ionic_strength * 1000

    # Calculate Debye screening parameter kappa
    kappa = math.sqrt((8 * math.pi * N_A * e**2 * ionic_strength) / (1000 * k_B * T))

    return kappa


def debye_screening_length(ionic_strength: float, T: int) -> float:
    """Calculate the Debye screening length in electrolyte.
    ISBN 978-0-444-63908-0

    Args:
        ionic_strength (float): Ionic strength of the solution in mol/L.
        T (int): Temperature in Kelvin.

    Returns:
        float: The Debye length kappa.
    """
    # Constants
    epsilon_0 = 8.854e-12  # Permittivity of free space in C^2/(Nm^2)

    # Temperature dependent relative permittivity of water
    epsilon_r = calculate_permittivity(T)

    N_A = 6.022e23  # Avogadro's number in mol^-1
    e = 1.602e-19  # Elementary charge in Coulombs
    k_B = 1.38e-23  # Boltzmann constant in J/K

    # Convert ionic strength to mol/m**3
    ionic_strength = ionic_strength * 1000

    # Calculate Debye length
    kappa = math.sqrt(
        (2 * N_A * e**2 * ionic_strength) / (epsilon_0 * epsilon_r * k_B * T)
    )

    return kappa
