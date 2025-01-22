from typing import Dict


def is_valid_peptide_sequence(pept: str) -> None:
    """
    Validate that the input is a valid peptide sequence.

    Args:
        seq (str): The input sequence.

    Raises:
        TypeError: If the input is not a string.
        ValueError: If the sequence contains invalid amino acids.
    """
    if not isinstance(pept, str):
        raise TypeError("Input must be a string.")

    if not len(pept) >= 6:
        raise ValueError("Sequence must be at least 6 amino acids long.")

    pept = pept.upper()
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids

    # check for invalid residues
    invalid_residues = [residue for residue in pept if residue not in valid_amino_acids]
    if invalid_residues:
        raise ValueError(
            f"Invalid residues found in sequence: {', '.join(invalid_residues)}"
        )


def is_valid_index(pept: str, i: int, j: int, min_helix_length: int, has_acetyl: bool, has_succinyl: bool, has_amide: bool) -> None:
    """
    Validate that the input indexes are valid.

    Args:
        pept (str): The peptide sequence.
        i (int): The start index.
        j (int): The helix length.
        min_helix_length (int): The minimum helix length.
        has_acetyl (bool): Whether the peptide has an N-terminal acetyl modification.
        has_succinyl (bool): Whether the peptide has a C-terminal succinyl modification.
        has_amide (bool): Whether the peptide has a C-terminal amide modification.
    Raises:
        TypeError: If the input indexes are not integers.
        ValueError: If the indexes are out of range.
    """
    if not isinstance(i, int) or not isinstance(j, int):
        raise TypeError("Indexes must be integers.")

    if i < 0:
        raise ValueError("Start index must be greater than or equal to zero.")

    if j < min_helix_length - (1 if has_acetyl or has_succinyl else 0) - (1 if has_amide else 0):
        raise ValueError(f"Helix length must be greater than or equal to {min_helix_length}.")

    if i + j > len(pept):
        raise ValueError(
            "The sum of the indexes must be less than the length of the sequence."
        )
