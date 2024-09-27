from typing import Dict


def is_valid_peptide_sequence(pept: str) -> None:
    """
    Validate that the input is a valid peptide sequence.

    Args:
        seq (str): The input sequence.

    Raises:
        TypeError: If the input is not a string.
        ValueError: If the sequence contains invalid amino acids or capping residues.
    """
    if not isinstance(pept, str):
        raise TypeError("Input must be a string.")
    
    if not len(pept) >= 6:
        raise ValueError("Sequence must be at least 6 amino acids long.")
    
    pept = pept.upper()
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY") # 20 standard amino acids
    valid_amino_acids.update(['Z', 'X', 'B']) # acetylation, succinylation, amidation
    
    # ckeck for invalid residues
    invalid_residues = [residue for residue in pept if residue not in valid_amino_acids]
    if invalid_residues:
        raise ValueError(f"Invalid residues found in sequence: {', '.join(invalid_residues)}")
    
    # get all indexes of Z and ensure they are at the beginning
    Z_indexes = [i for i, residue in enumerate(pept) if residue == 'Z']
    if len(set(Z_indexes) - set([0])) != 0:
        raise ValueError("Acetylation (Z) can only be at the beginning of the sequence.")
    
    # get all indexes of X and ensure they are at the beginning
    X_indexes = [i for i, residue in enumerate(pept) if residue == 'X']
    if len(set(X_indexes) - set([0])) != 0:
        raise ValueError("Succinylation (X) can only be at the beginning of the sequence.")
    
    # get all indexes of B and ensure they are at the end
    B_indexes = [i for i, residue in enumerate(pept) if residue == 'B']
    if len(set(B_indexes) - set([len(pept)-1])) != 0:
        raise ValueError("Amidation (B) can only be at the end of the sequence.")


def is_valid_index(pept: str, i: int, j: int) -> None:
    """
    Validate that the input indexes are valid.

    Args:
        pept (str): The peptide sequence.
        i (int): The start index.
        j (int): The helix length.

    Raises:
        TypeError: If the input indexes are not integers.
        ValueError: If the indexes are out of range.
    """
    if not isinstance(i, int) or not isinstance(j, int):
        raise TypeError("Indexes must be integers.")
    
    if i < 0:
        raise ValueError("Start index must be greater than or equal to zero.")
    
    if j < 6:
        raise ValueError("Helix length must be greater than or equal to 6.")
   
    if i + j > len(pept):
        raise ValueError("The sum of the indexes must be less than the length of the sequence.")
    

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
        "Na+": 1, "K+": 1, "Li+": 1, "NH4+": 1,
        
        # Divalent cations
        "Ca2+": 2, "Mg2+": 2, "Mn2+": 2, "Fe2+": 2, "Zn2+": 2, "Cu2+": 2,
        
        # Monovalent anions
        "Cl-": -1, "Br-": -1, "I-": -1, "F-": -1, "NO3-": -1, "HCO3-": -1,
        
        # Divalent anions
        "SO42-": -2, "HPO42-": -2,
        
        # Trivalent anions
        "PO43-": -3,

    }
    
    I = 0.5 * sum(conc * ion_charges[ion]**2 for ion, conc in ion_concentrations.items() if ion in ion_charges)
    
    # Check for any unrecognized ions
    unrecognized = [ion for ion in ion_concentrations if ion not in ion_charges]
    if unrecognized:
        print(f"Warning: The following ions were not recognized and were not included in the calculation: {', '.join(unrecognized)}")
    
    return I