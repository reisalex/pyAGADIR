import pytest
from pyagadir.models import AGADIR
from pyagadir.models import calc_K, get_dG_dipole, get_dG_Hbond
import numpy as np
import pandas as pd
from importlib.resources import files

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


def get_hprob_from_energy(energy):
    """Get the helix probability from the energy.
    """
    return calc_K(energy) / (1 + calc_K(energy))


def test_dG_Int():
    """Test the intrinsic propensities.
    """
    # ensure that the intrinsic propensities match those in Table2, case with single AA repeats
    pep_len = 6
    for method in ['r', '1s']:
        for aa in table2.index:
            model = AGADIR(method=method)
            result = model.predict(aa*pep_len)
            print(aa)
            print(result.int_array)
            assert all([result.int_array[i] == table2.loc[aa, 'Intrinsic'] for i in range(pep_len)])
            assert result.int_array.shape == (pep_len,)

    # ensure that the intrinsic propensities match those in Table2, case with the 20 AA
    for method in ['r', '1s']:
        pept = ''.join(table2.index)
        model = AGADIR(method=method)
        result = model.predict(pept)
        assert all([result.int_array[i] == table2.loc[aa, 'Intrinsic'] for i, aa in enumerate(pept)])
        assert result.int_array.shape == (20,)


def test_dG_SD():
    """Test the side chain interactions.
    """
    # no side chain interactions for the case of only Ala
    for method in ['r', '1s']:
        pep = 'AAAAAAAAAAAA'
        model = AGADIR(method=method)
        result = model.predict(pep)
        assert all([result.i1_array[i] == 0.0 for i in range(len(pep))])
        assert all([result.i3_array[i] == 0.0 for i in range(len(pep))])
        assert all([result.i4_array[i] == 0.0 for i in range(len(pep))])

    # C to S interaction
    for method in ['r', '1s']:
        pep = 'ACAACSAAAAAA'
        model = AGADIR(method=method)
        result = model.predict(pep)
        assert result.i3_array[1] == 0.2 # C to C interaction
        assert result.i4_array[1] == 0.2 # C to S interaction
        assert all([result.i1_array[i] == 0.0 for i in range(len(pep))])
        assert all([result.i3_array[i] == 0.0 for i in range(len(pep)) if i != 1])
        assert all([result.i4_array[i] == 0.0 for i in range(len(pep)) if i != 1])    

    # M to W,  M to F and W to M interaction
    for method in ['r', '1s']:
        pep = 'AMAAWFAMAAAA'
        model = AGADIR(method=method)
        result = model.predict(pep)
        assert result.i3_array[1] == -0.3 # M to W interaction
        assert result.i4_array[1] == -0.2 # M to F interaction
        assert result.i3_array[4] == -0.25 # W to M interaction
        assert all([result.i1_array[i] == 0.0 for i in range(len(pep))])
        assert all([result.i3_array[i] == 0.0 for i in range(len(pep)) if i not in [1, 4]])
        assert all([result.i4_array[i] == 0.0 for i in range(len(pep)) if i not in [1, 4]])

    # bad interaction with negative charge at position i + 1, 3, 4
    for method in ['r', '1s']:
        pep = 'ADAADDAAAAAA'
        model = AGADIR(method=method)
        result = model.predict(pep)
        assert result.i1_array[4] == 0.05
        assert result.i3_array[1] == 0.1
        assert result.i4_array[1] == 0.2
        assert all([result.i1_array[i] == 0.0 for i in range(len(pep)) if i != 4])
        assert all([result.i3_array[i] == 0.0 for i in range(len(pep)) if i != 1])
        assert all([result.i4_array[i] == 0.0 for i in range(len(pep)) if i != 1])

    # bad interaction with positive charge at position i + 1, 3, 4
    for method in ['r', '1s']:
        pep = 'AKAAKKAAAAAA'
        model = AGADIR(method=method)
        result = model.predict(pep)
        assert result.i1_array[4] == 0.05
        assert result.i3_array[1] == 0.25
        assert result.i4_array[1] == 0.2
        assert all([result.i1_array[i] == 0.0 for i in range(len(pep)) if i != 4])
        assert all([result.i3_array[i] == 0.0 for i in range(len(pep)) if i != 1])
        assert all([result.i4_array[i] == 0.0 for i in range(len(pep)) if i != 1])

    # good interaction with opposite charge at position i + 1, 3, 4
    for method in ['r', '1s']:
        pep = 'AKAAEAAAKDAA'
        model = AGADIR(method=method)
        result = model.predict(pep)
        assert result.i1_array[8] == -0.05
        assert result.i3_array[1] == -0.1
        assert result.i4_array[4] == -0.33
        assert all([result.i1_array[i] == 0.0 for i in range(len(pep)) if i != 8])
        assert all([result.i3_array[i] == 0.0 for i in range(len(pep)) if i != 1])
        assert all([result.i4_array[i] == 0.0 for i in range(len(pep)) if i != 4])


def test_dG_nonH():
    """Test the capping interactions.
    """
    ### N capping ###
    # need to better understand the N capping interactions before writing additional tests

    ### C capping ###
    # Ala has no capping interactions
    for method in ['r', '1s']:
        pep = 'AAAAAAAAAAAA'
        model = AGADIR(method=method)
        result = model.predict(pep)
        assert all([result.C_array[i] == 0.0 for i in range(len(pep))])

    # there are 3 Cys, but only the last two should have C-capping interactions
    for method in ['r', '1s']:
        pep = 'AACAACAAAAAC'
        model = AGADIR(method=method)
        result = model.predict(pep)
        assert result.C_array[5] == 0.08
        assert result.C_array[11] == 0.08
        assert all([result.C_array[i] == 0.0 for i in range(len(pep)) if i not in [5, 11]])

    # test Ser and His (first five should not have C-capping interactions)
    for method in ['r', '1s']:
        pep = 'MREWQAASAAHA'
        model = AGADIR(method=method)
        result = model.predict(pep)
        assert result.C_array[7] == -0.08
        assert result.C_array[10] == -0.35
        assert all([result.C_array[i] == 0.0 for i in range(len(pep)) if i not in [7, 10]])


def test_dG_dipole():
    """Test the dipole moments.
    """
    # Ala does not influence dipole moment, so use it as a base to test the effect of amino acids.
    # The effect should be as described in Table3a and Table3b in the data folder.

    # test the C-terminal
    n_energy, c_energy = get_dG_dipole('AAAAAAAAAAH')
    assert (sum(n_energy) == 0.0 and sum(c_energy) == -0.44)
    assert c_energy[-1] == -0.44

    n_energy, c_energy = get_dG_dipole('AAAAAAAAAHA')
    assert (sum(n_energy) == 0.23 and sum(c_energy) == -0.34)

    # test the N-terminal
    n_energy, c_energy = get_dG_dipole('DAAAAAAAAAA')
    assert (sum(n_energy) == -0.34 and sum(c_energy) == 0.0)
    assert n_energy[0] == -0.34

    n_energy, c_energy = get_dG_dipole('ADAAAAAAAAA')
    assert (sum(n_energy) == -0.51 and sum(c_energy) == 0.08)

    # check all values
    n_energy, c_energy = get_dG_dipole('DDDDDDDDDDD')
    assert (n_energy == [-0.34, -0.51, -0.53, -0.42, -0.18, -0.15, -0.13, -0.12, -0.11, -0.09, 0.0])
    assert (c_energy == [0.0, 0.08, 0.1, 0.13, 0.17, 0.18, 0.22, 0.26, 0.53, 0.9, 0.58])

    n_energy, c_energy = get_dG_dipole('EEEEEEEEEEE')
    assert (n_energy == [-0.26, -0.39, -0.22, -0.19, -0.18, -0.15, -0.13, -0.12, -0.09, -0.08, 0.0])
    assert (c_energy == [0.0, 0.07, 0.09, 0.1, 0.13, 0.15, 0.16, 0.2, 0.38, 0.46, 0.4])

    n_energy, c_energy = get_dG_dipole('HHHHHHHHHHH')
    assert (n_energy == [0.33, 1.4, 1.4, 0.52, 0.39, 0.36, 0.34, 0.32, 0.26, 0.23, 0.0])
    assert (c_energy == [0.0, -0.08, -0.09, -0.09, -0.09, -0.11, -0.13, -0.19, -0.23, -0.34, -0.44])

    n_energy, c_energy = get_dG_dipole('KKKKKKKKKKK')
    assert (n_energy == [0.43, 0.38, 0.64, 0.58, 0.48, 0.27, 0.24, 0.2, 0.18, 0.11, 0.0])
    assert (c_energy == [0.0, -0.08, -0.09, -0.1, -0.12, -0.13, -0.26, -0.32, -0.34, -0.36, -0.51])

    n_energy, c_energy = get_dG_dipole('RRRRRRRRRRR')
    assert (n_energy == [0.29, 0.33, 0.44, 0.4, 0.34, 0.21, 0.19, 0.16, 0.15, 0.09, 0.0])
    assert (c_energy == [0.0, -0.07, -0.07, -0.09, -0.09, -0.11, -0.25, -0.24, -0.26, -0.27, -0.36])


def test_dG_Hbond():
    """Test the hydrogen bonding interactions.
    """
    pept = 'AAAAAAAAAAH'
    h_energy = get_dG_Hbond(len(pept))
    assert h_energy == -0.775 * max((len(pept) - 6), 0)

    pept = 'AAHAAH'
    h_energy = get_dG_Hbond(len(pept))
    assert h_energy == 0.0

