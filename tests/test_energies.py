import pytest
from pyagadir.models import AGADIR
from pyagadir.models import calc_K
from pyagadir.models import get_dG_dipole, get_dG_Int, get_dG_i1, get_dG_i3, get_dG_i4, get_dG_Ncap, get_dG_Ccap, get_dG_Hbond

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
    for aa in table2.index:
        pept = aa * pep_len
        int_array = get_dG_Int(pept)
        assert int_array.shape[0] == len(pept)
        assert int_array[0] == 0.0
        assert int_array[-1] == 0.0
        assert all([0.66 if (i == 1 and aa == 'P') else int_array[i] == table2.loc[aa, 'Intrinsic'] 
                    for i in range(1, pep_len-1)])

    # ensure that the intrinsic propensities match those in Table2, case with the 20 AA
    pept = ''.join(table2.index)
    int_array = get_dG_Int(pept)
    assert all([int_array[i] == 0.0 if (i == 0 or i == len(pept)-1) else int_array[i] == table2.loc[aa, 'Intrinsic'] 
                for i, aa in enumerate(pept)])
    assert int_array.shape[0] == 20


def test_dG_SD():
    """Test the side chain interactions.
    """
    # no side chain interactions for the case of only Ala
    pept = 'AAAAAAAAAAAA'
    i1_array = get_dG_i1(pept)
    i3_array = get_dG_i3(pept)
    i4_array = get_dG_i4(pept)
    assert i1_array.shape[0] == len(pept)
    assert i3_array.shape[0] == len(pept)
    assert i4_array.shape[0] == len(pept)
    assert all([i1_array[i] == 0.0 for i in range(len(pept))])
    assert all([i3_array[i] == 0.0 for i in range(len(pept))])
    assert all([i4_array[i] == 0.0 for i in range(len(pept))])

    # C to S interaction
    pept = 'ACAACSAAAAAA'
    i1_array = get_dG_i1(pept)
    i3_array = get_dG_i3(pept)
    i4_array = get_dG_i4(pept)
    assert i3_array[1] == 0.2 # C to C interaction
    assert i4_array[1] == 0.2 # C to S interaction
    assert all([i1_array[i] == 0.0 for i in range(len(pept))])
    assert all([i3_array[i] == 0.0 for i in range(len(pept)) if i != 1])
    assert all([i4_array[i] == 0.0 for i in range(len(pept)) if i != 1])    

    # M to W,  M to F and W to M interaction
    pept = 'AMAAWFAMAAAA'
    i1_array = get_dG_i1(pept)
    i3_array = get_dG_i3(pept)
    i4_array = get_dG_i4(pept)
    assert i3_array[1] == -0.3 # M to W interaction
    assert i4_array[1] == -0.2 # M to F interaction
    assert i3_array[4] == -0.25 # W to M interaction
    assert all([i1_array[i] == 0.0 for i in range(len(pept))])
    assert all([i3_array[i] == 0.0 for i in range(len(pept)) if i not in [1, 4]])
    assert all([i4_array[i] == 0.0 for i in range(len(pept)) if i not in [1, 4]])

    # bad interaction with negative charge at position i + 1, 3, 4
    pept = 'ADAADDAAAAAA'
    i1_array = get_dG_i1(pept)
    i3_array = get_dG_i3(pept)
    i4_array = get_dG_i4(pept)
    assert i1_array[4] == 0.05
    assert i3_array[1] == 0.1
    assert i4_array[1] == 0.2
    assert all([i1_array[i] == 0.0 for i in range(len(pept)) if i != 4])
    assert all([i3_array[i] == 0.0 for i in range(len(pept)) if i != 1])
    assert all([i4_array[i] == 0.0 for i in range(len(pept)) if i != 1])

    # bad interaction with positive charge at position i + 1, 3, 4
    pept = 'AKAAKKAAAAAA'
    i1_array = get_dG_i1(pept)
    i3_array = get_dG_i3(pept)
    i4_array = get_dG_i4(pept)
    assert i1_array[4] == 0.05
    assert i3_array[1] == 0.25
    assert i4_array[1] == 0.2
    assert all([i1_array[i] == 0.0 for i in range(len(pept)) if i != 4])
    assert all([i3_array[i] == 0.0 for i in range(len(pept)) if i != 1])
    assert all([i4_array[i] == 0.0 for i in range(len(pept)) if i != 1])

    # good interaction with opposite charge at position i + 1, 3, 4
    pept = 'AKAAEAAAKDAA'
    i1_array = get_dG_i1(pept)
    i3_array = get_dG_i3(pept)
    i4_array = get_dG_i4(pept)
    assert i1_array[8] == -0.05
    assert i3_array[1] == -0.1
    assert i4_array[4] == -0.33
    assert all([i1_array[i] == 0.0 for i in range(len(pept)) if i != 8])
    assert all([i3_array[i] == 0.0 for i in range(len(pept)) if i != 1])
    assert all([i4_array[i] == 0.0 for i in range(len(pept)) if i != 4])


def test_dG_nonH():
    """Test the capping interactions.
    """
    ### N capping ###
    # need to better understand the N capping interactions before writing additional tests

    ### C capping ###
    # Ala has no capping interactions
    pept = 'AAAAAAAAAAA'
    c_cap = get_dG_Ccap(pept)
    assert c_cap == 0.0

    # there are 3 Cys, but only first should have C-capping interactions
    pept = 'CAAAACAAAAAC'
    c_cap = get_dG_Ccap(pept)
    assert c_cap== 0.08

    # M does not have C-capping interactions)
    pept = 'MREWQAASAAHA'
    c_cap = get_dG_Ccap(pept)
    assert c_cap == 0.0


def test_dG_dipole():
    """Test the dipole moments.
    """
    # Ala does not influence dipole moment, so use it as a base to test the effect of amino acids.
    # The effect should be as described in Table3a and Table3b in the data folder.

    # test the C-terminal
    n_energy, c_energy = get_dG_dipole('AAAAAAAAAAH')
    assert (n_energy.shape[0] == len('AAAAAAAAAAH'))
    assert (c_energy.shape[0] == len('AAAAAAAAAAH'))
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
    assert np.all(n_energy == [-0.34, -0.51, -0.53, -0.42, -0.18, -0.15, -0.13, -0.12, -0.11, -0.09, 0.0])
    assert np.all(c_energy == [0.0, 0.08, 0.1, 0.13, 0.17, 0.18, 0.22, 0.26, 0.53, 0.9, 0.58])

    n_energy, c_energy = get_dG_dipole('EEEEEEEEEEE')
    assert np.all(n_energy == [-0.26, -0.39, -0.22, -0.19, -0.18, -0.15, -0.13, -0.12, -0.09, -0.08, 0.0])
    assert np.all(c_energy == [0.0, 0.07, 0.09, 0.1, 0.13, 0.15, 0.16, 0.2, 0.38, 0.46, 0.4])

    n_energy, c_energy = get_dG_dipole('HHHHHHHHHHH')
    assert np.all(n_energy == [0.33, 1.4, 1.4, 0.52, 0.39, 0.36, 0.34, 0.32, 0.26, 0.23, 0.0])
    assert np.all(c_energy == [0.0, -0.08, -0.09, -0.09, -0.09, -0.11, -0.13, -0.19, -0.23, -0.34, -0.44])

    n_energy, c_energy = get_dG_dipole('KKKKKKKKKKK')
    assert np.all(n_energy == [0.43, 0.38, 0.64, 0.58, 0.48, 0.27, 0.24, 0.2, 0.18, 0.11, 0.0])
    assert np.all(c_energy == [0.0, -0.08, -0.09, -0.1, -0.12, -0.13, -0.26, -0.32, -0.34, -0.36, -0.51])

    n_energy, c_energy = get_dG_dipole('RRRRRRRRRRR')
    assert np.all(n_energy == [0.29, 0.33, 0.44, 0.4, 0.34, 0.21, 0.19, 0.16, 0.15, 0.09, 0.0])
    assert np.all(c_energy == [0.0, -0.07, -0.07, -0.09, -0.09, -0.11, -0.25, -0.24, -0.26, -0.27, -0.36])


def test_dG_Hbond():
    """Test the hydrogen bonding interactions.
    """
    pept = 'AAAAAAAAAAH'
    h_energy = get_dG_Hbond(pept)
    assert h_energy == -0.775 * max((len(pept) - 6), 0)

    pept = 'AAHAAH'
    h_energy = get_dG_Hbond(pept)
    assert h_energy == 0.0

