import pytest
from pyagadir import predict_alphahelix
from pyagadir.models import ModelResult
from pyagadir.models import AGADIR
import numpy as np

# This is the function we will be testing
def add(x, y):
    return x + y


# Test bad inputs
def test_bad_input():
    # wrong type
    with pytest.raises(ValueError):
        result = predict_alphahelix(5)

    # wrong amino acids
    with pytest.raises(ValueError):
        result = predict_alphahelix('Munoz rocks!')
    

# Test simple prediction case
def test_simple_predict():

    # predict
    result = predict_alphahelix('ILKSLEEFLKVTLRSTRQT')

    # Check the result
    assert isinstance(result, ModelResult)
    assert result.seq == 'ILKSLEEFLKVTLRSTRQT'
    assert len(result.helical_propensity) == 19

    assert isinstance(result.get_helical_propensity(), np.ndarray)
    assert result.get_helical_propensity().shape == (19,)

    assert isinstance(result.helical_propensity[0], float)
    assert abs(result.helical_propensity[0] - 0.00734307) < 1e-6

    assert isinstance(result.get_percent_helix(), float)
    assert 1.0 >= result.get_percent_helix() >= 0.0


# Test the more custom case
def test_custom_predict():
    # test bad input
    with pytest.raises(ValueError):
        model = AGADIR(method='XYZ')
    
    # test good inputs
    model = AGADIR(method='r')
    assert isinstance(model, AGADIR)
    model = AGADIR(method='1s') # this is the default
    assert isinstance(model, AGADIR)

    # test prediction
    model = AGADIR() # 1s is the default, otherwise value tests fail at the end of the tests
    assert isinstance(model, AGADIR)
    result = model.predict('ILKSLEEFLKVTLRSTRQT')
    assert isinstance(result, ModelResult)


    # Check the result
    assert result.seq == 'ILKSLEEFLKVTLRSTRQT'

    assert isinstance(result.int_array, np.ndarray)
    assert result.int_array.shape == (19,)

    assert isinstance(result.i1_array, np.ndarray)
    assert result.i1_array.shape == (19,)

    assert isinstance(result.i3_array, np.ndarray)
    assert result.i3_array.shape == (19,)

    assert isinstance(result.i4_array, np.ndarray)
    assert result.i4_array.shape == (19,)

    assert isinstance(result.N_array, np.ndarray)
    assert result.N_array.shape == (19,)

    assert isinstance(result.C_array, np.ndarray)
    assert result.C_array.shape == (19,)

    assert isinstance(result.dG_dict_mat, list)
    assert len(result.dG_dict_mat) == 20   # why 20???? 

    assert isinstance(result.dG_dict_mat[0], type(None)) # why is this None? README says this should be a list of lists
    assert isinstance(result.dG_dict_mat[1], type(None)) # why is this None? README says this should be a list of lists

    assert isinstance(result.K_tot, float)

    assert isinstance(result.K_tot_array, np.ndarray)
    assert result.K_tot_array.shape == (19,)

    assert isinstance(result.Z, float)

    assert isinstance(result.Z_array, np.ndarray)
    assert result.Z_array.shape == (19,)

    assert isinstance(result.helical_propensity, np.ndarray)
    assert result.helical_propensity.shape == (19,)

    assert isinstance(result.percent_helix, float)

    # test the get functions
    assert isinstance(result.get_helical_propensity(), np.ndarray)
    assert result.get_helical_propensity().shape == (19,)

    assert isinstance(result.get_percent_helix(), float)

    # test the values
    assert abs(result.get_helical_propensity()[0] - 0.00734307) < 1e-6
    assert 1.0 >= result.get_percent_helix() >= 0.0
    assert abs(result.get_percent_helix() - 0.092) < 1e-6



# add test using good and bade helices
    
