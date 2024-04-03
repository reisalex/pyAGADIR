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
    

# # Test simple prediction case
# def test_simple_predict():

#     # predict
#     result = predict_alphahelix('ILKSLEEFLKVTLRSTRQT')

#     # Check the result
#     assert isinstance(result, ModelResult)
#     assert result.seq == 'ILKSLEEFLKVTLRSTRQT'
#     assert len(result.helical_propensity) == 19

#     assert isinstance(result.get_helical_propensity(), np.ndarray)
#     assert result.get_helical_propensity().shape == (19,)

#     assert isinstance(result.helical_propensity[0], float)
#     assert abs(result.helical_propensity[0] - 0.00734307) < 1e-6

#     assert isinstance(result.get_percent_helix(), float)
#     assert 1.0 >= result.get_percent_helix() >= 0.0


# # Test the more custom case
# def test_custom_predict():
#     # test bad input
#     with pytest.raises(ValueError):
#         model = AGADIR(method='XYZ')
    
#     # test good inputs
#     model = AGADIR(method='r')
#     assert isinstance(model, AGADIR)
#     model = AGADIR(method='1s') # this is the default
#     assert isinstance(model, AGADIR)

#     # test prediction
#     model = AGADIR() # 1s is the default, otherwise value tests fail at the end of the tests
#     assert isinstance(model, AGADIR)
#     result = model.predict('ILKSLEEFLKVTLRSTRQT')
#     assert isinstance(result, ModelResult)


#     # Check the result
#     assert result.seq == 'ILKSLEEFLKVTLRSTRQT'

#     assert isinstance(result.int_array, np.ndarray)
#     assert result.int_array.shape == (19,)

#     assert isinstance(result.i1_array, np.ndarray)
#     assert result.i1_array.shape == (19,)

#     assert isinstance(result.i3_array, np.ndarray)
#     assert result.i3_array.shape == (19,)

#     assert isinstance(result.i4_array, np.ndarray)
#     assert result.i4_array.shape == (19,)

#     assert isinstance(result.N_array, np.ndarray)
#     assert result.N_array.shape == (19,)

#     assert isinstance(result.C_array, np.ndarray)
#     assert result.C_array.shape == (19,)

#     assert isinstance(result.dG_dict_mat, list)
#     assert len(result.dG_dict_mat) == 20   # why 20???? 

#     assert isinstance(result.dG_dict_mat[0], type(None)) # why is this None? README says this should be a list of lists
#     assert isinstance(result.dG_dict_mat[1], type(None)) # why is this None? README says this should be a list of lists

#     assert isinstance(result.K_tot, float)

#     assert isinstance(result.K_tot_array, np.ndarray)
#     assert result.K_tot_array.shape == (19,)

#     assert isinstance(result.Z, float)

#     assert isinstance(result.Z_array, np.ndarray)
#     assert result.Z_array.shape == (19,)

#     assert isinstance(result.helical_propensity, np.ndarray)
#     assert result.helical_propensity.shape == (19,)

#     assert isinstance(result.percent_helix, float)

#     # test the get functions
#     assert isinstance(result.get_helical_propensity(), np.ndarray)
#     assert result.get_helical_propensity().shape == (19,)

#     assert isinstance(result.get_percent_helix(), float)

#     # test the values
#     assert abs(result.get_helical_propensity()[0] - 0.00734307) < 1e-6
#     assert 1.0 >= result.get_percent_helix() >= 0.0
#     assert abs(result.get_percent_helix() - 0.092) < 1e-6



# test using data from papers
def test_paper_data_figure_3():
    # These were extracted from Figure 3 in https://doi.org/10.1002/(SICI)1097-0282(19970415)41:5<495::AID-BIP2>3.0.CO;2-H
    # the values are approximate, expect some error
    x_vals_fig_3 = [1, 2, 4, 6, 7, 9, 10, 12, 14, 15]
    y_vals_fig_3 = [54.9, 46.6, 44.3, 35.5, 46.5, 26.5, 26.5, 36.5, 28.6, 34.9]

    # The peptide sequences come from Table 1 in https://doi.org/10.1002/pro.5560021006 (with original data in Fig 1B)
    # Represent Acetyl with Z, succinyl with X, and amide with B
    # Data originally from huyghues-despointes et al. (1993)

    peptides = ['ZDAQAAAAQAAAAQAAYB',
                'ZADQAAAAQAAAAQAAYB',
                'ZAAQDAAAQAAAAQAAYB',
                'ZAAQAADAQAAAAQAAYB',
                'ZAAQAAADQAAAAQAAYB',
                'ZAAQAAAAQDAAAQAAYB',
                'ZAAQAAAAQADAAQAAYB',
                'ZAAQAAAAQAAADQAAYB',
                'ZAAQAAAAQAAAAQDAYB',
                'ZAAQAAAAQAAAAQADYB',]
    tol = 3.0

    for i, pept in enumerate(peptides):
        model = AGADIR(method='1s', T=4.0)
        result = model.predict(pept)
        assert abs(result.get_percent_helix()*100 - y_vals_fig_3[i]) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {y_vals_fig_3[i]}, {pept}, method: 1s"


# test using data from papers
def test_paper_data_figure_4A():
    # These were extracted from Figure 4 in https://doi.org/10.1002/(SICI)1097-0282(19970415)41:5<495::AID-BIP2>3.0.CO;2-H
    # the values are approximate, expect some error.
    # Original work from Scholtz et al. (1991)a

    # Represent Acetyl with Z, succinyl with X, and amide with B

    # Figure 4A, AAQAA repeats, both methods should give comparable results (using the AGADIRms values becaus the paper states that 1s and ms are similar)
    y_vals_fig_4A = [0.0,
                     15.7,
                     41.3,
                     61.3]
    peptides_fig_4A = ['ZAAQAAYB',
                       'ZAAQAAAAQAAYB',
                       'ZAAQAAAAQAAAAQAAYB',
                       'ZAAQAAAAQAAAAQAAAAQAAYB']
    
    tol = 3.0

    for i, pept in enumerate(peptides_fig_4A):
        model = AGADIR(method='1s', T=4.0)
        result = model.predict(pept)
        assert abs(result.get_percent_helix()*100 - y_vals_fig_4A[i]) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {y_vals_fig_4A[i]}, {pept}, method: 1s"

    for i, pept in enumerate(peptides_fig_4A):
        model = AGADIR(method='r', T=4.0)
        result = model.predict(pept)
        assert abs(result.get_percent_helix()*100 - y_vals_fig_4A[i]) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {y_vals_fig_4A[i]}, {pept}, method: r"


def test_paper_data_figure_4B():
    # These were extracted from Figure 4 in https://doi.org/10.1002/(SICI)1097-0282(19970415)41:5<495::AID-BIP2>3.0.CO;2-H
    # the values are approximate, expect some error.
    # Original work from Rohl et al. (1992)
    # These are: Ac-(AAKAA)nY-NH2

    # Represent Acetyl with Z, succinyl with X, and amide with B

    # Figure 4B, AAKAA repeats, AGADIR should predict more helix than AGADIR1s (using the AGADIRms values becaus the paper states that 1s and ms are similar)
    y_vals_fig_4B_r = [51.8,
                       80.7,
                       94.0,
                       100.6,
                       100.5]
    y_vals_fig_4B_1s = [50.5,
                        71.4,
                        80.7,
                        89.4,
                        91.7]
    peptides_fig_4B = ['ZAAKAAAAKAAAAKAAYB',
                       'ZAAKAAAAKAAAAKAAAAKAAYB',
                       'ZAAKAAAAKAAAAKAAAAKAAAAKAAYB',
                       'ZAAKAAAAKAAAAKAAAAKAAAAKAAAAKAAAAKAAAAKAAYB',
                       'ZAAKAAAAKAAAAKAAAAKAAAAKAAAAKAAAAKAAAAKAAAAKAAAAKAAYB']

    tol = 3.0

    for i, pept in enumerate(peptides_fig_4B):
        model = AGADIR(method='1s', T=4.0)
        result = model.predict(pept)
        assert abs(result.get_percent_helix()*100 - y_vals_fig_4B_1s[i]) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {y_vals_fig_4B_1s[i]}, {pept}, method: 1s"

    for i, pept in enumerate(peptides_fig_4B):
        model = AGADIR(method='r', T=4.0)
        result = model.predict(pept)
        assert abs(result.get_percent_helix()*100 - y_vals_fig_4B_r[i]) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {y_vals_fig_4B_r[i]}, {pept}, method: r"


def test_paper_data_figure_4C():
    # These were extracted from Figure 4 in https://doi.org/10.1002/(SICI)1097-0282(19970415)41:5<495::AID-BIP2>3.0.CO;2-H
    # the values are approximate, expect some error.
    # Original work from Scholtz et al. (1991)b

    # Figure 4C, AEAAKA repeats
    y_vals_fig_4C_r = [48.0,
                       79.2,
                       94.1,
                       100.1,
                       100.5]
    y_vals_fig_4C_1s = [48.3,
                        72.7,
                        82.6,
                        89.8,
                        92.8]
    peptides_fig_4C = ['ZYAEAAKAAEAAKAAEAAKAFB',
                       'ZYAEAAKAAEAAKAAEAAKAAEAAKAFB',
                       'ZYAEAAKAAEAAKAAEAAKAAEAAKAAEAAKAFB',
                       'ZYAEAAKAAEAAKAAEAAKAAEAAKAAEAAKAAEAAKAAEAAKAAEAAKAFB',
                       'ZYAEAAKAAEAAKAAEAAKAAEAAKAAEAAKAAEAAKAAEAAKAAEAAKAAEAAKAAEAAKAFB']
    
    tol = 3.0

    for i, pept in enumerate(peptides_fig_4C):
        model = AGADIR(method='1s', T=4.0)
        result = model.predict(pept)
        assert abs(result.get_percent_helix()*100 - y_vals_fig_4C_1s[i]) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {y_vals_fig_4C_1s[i]}, {pept}, method: 1s"

    for i, pept in enumerate(peptides_fig_4C):
        model = AGADIR(method='r', T=4.0)
        result = model.predict(pept)
        assert abs(result.get_percent_helix()*100 - y_vals_fig_4C_r[i]) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {y_vals_fig_4C_r[i]}, {pept}, method: r"



def test_paper_2_table_1_CheY2():
    """Data from
    https://doi.org/10.1006/jmbi.1994.0023

    The values are approximate, expect some error.
    """

    tol = 3.0

    # CheY2
    pept='EDGVDALNKLQAGGY'
    pred=4
    model = AGADIR(method='1s', T=4.0)
    result = model.predict(pept)
    assert abs(result.get_percent_helix()*100 - pred) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {pred}, {pept}, method: 1s"

    #CheY2-Mo
    pept='EDAVEALRKLQAGGY'
    pred=40
    model = AGADIR(method='1s', T=4.0)
    result = model.predict(pept)
    assert abs(result.get_percent_helix()*100 - pred) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {pred}, {pept}, method: 1s"


def test_paper_2_table_1_CheY5():
    """Data from
    https://doi.org/10.1006/jmbi.1994.0023

    The values are approximate, expect some error.
    """

    tol = 3.0

    # CheY5
    pept='AATLEEKLNKIFEKLGMY'
    pred=7
    model = AGADIR(method='1s', T=4.0)
    result = model.predict(pept)
    assert abs(result.get_percent_helix()*100 - pred) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {pred}, {pept}, method: 1s"

    # CheY5-Mo
    pept='AATLAEKLAKILEKLGGY'
    pred=20
    model = AGADIR(method='1s', T=4.0)
    result = model.predict(pept)
    assert abs(result.get_percent_helix()*100 - pred) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {pred}, {pept}, method: 1s"


def test_paper_2_table_1_SH3Lo():
    """Data from
    https://doi.org/10.1006/jmbi.1994.0023

    The values are approximate, expect some error.
    """

    tol = 3.0

    # SH3Lo
    pept='DYQEKSPREVAMKKG'
    pred=2
    model = AGADIR(method='1s', T=4.0)
    result = model.predict(pept)
    assert abs(result.get_percent_helix()*100 - pred) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {pred}, {pept}, method: 1s"

    # Sh3Lo-Mo
    pept='TYQEKAAREVAMKKG'
    pred=15
    model = AGADIR(method='1s', T=4.0)
    result = model.predict(pept)
    assert abs(result.get_percent_helix()*100 - pred) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {pred}, {pept}, method: 1s"


def test_paper_3_figure_2A():
    """Data from
    https://doi.org/10.1006/jmbi.1994.0024
    
    Only used first data point, triangles.
    The values are approximate, expect some error.
    """
    tol = 3.0
    pept = 'ZAAQAAAAQAAAAQAAYB'
    pred = 44.1
    model = AGADIR(method='1s', T=0.0)
    result = model.predict(pept)
    assert abs(result.get_percent_helix()*100 - pred) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {pred}, {pept}, method: 1s"


def test_paper_3_figure_2B():
    """Data from
    https://doi.org/10.1006/jmbi.1994.0024
    
    Only used first data point, triangles.
    The values are approximate, expect some error.
    """
    tol = 3.0
    pept = 'ZAAAAKAAAAKAAAAKAB'
    pred = 61.8
    model = AGADIR(method='1s', T=0.0)
    result = model.predict(pept)
    assert abs(result.get_percent_helix()*100 - pred) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {pred}, {pept}, method: 1s"


def test_paper_3_figure_2C():
    """Data from
    https://doi.org/10.1006/jmbi.1994.0024
    
    Only used first data point, triangles.
    The values are approximate, expect some error.
    """
    tol = 3.0
    pept = 'ZAEAAAKEAAAKEAAAKAB'
    pred = 90.6
    model = AGADIR(method='1s', T=0.0)
    result = model.predict(pept)
    assert abs(result.get_percent_helix()*100 - pred) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {pred}, {pept}, method: 1s"


def test_paper_3_figure_2D():
    """Data from
    https://doi.org/10.1006/jmbi.1994.0024
    
    Only used first data point, triangles.
    The values are approximate, expect some error.
    """
    tol = 3.0
    pept = 'ZWDAAAKDAAAKDAAAKAB'
    pred = 65.0
    model = AGADIR(method='1s', T=0.0)
    result = model.predict(pept)
    assert abs(result.get_percent_helix()*100 - pred) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {pred}, {pept}, method: 1s"


def test_paper_3_figure_2E():
    """Data from
    https://doi.org/10.1006/jmbi.1994.0024
    
    Only used first data point, triangles.
    The values are approximate, expect some error.
    """
    tol = 3.0
    pept = 'ZEAEKAAKEAEKAAKEAEKB'
    pred = 56.4
    model = AGADIR(method='1s', T=4.0) # yes, this one is 4.0, not 0.0
    result = model.predict(pept)
    assert abs(result.get_percent_helix()*100 - pred) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {pred}, {pept}, method: 1s"


def test_paper_3_figure_3A():
    """Data from
    https://doi.org/10.1006/jmbi.1994.0024
    
    Only used first data point, triangles.
    The values are approximate, expect some error.
    """
    tol = 3.0
    pept = 'ZGADAQGAMNKALELFRKDIAAKYKELGB'
    pred = 37.4
    model = AGADIR(method='1s', T=0.0)
    result = model.predict(pept)
    assert abs(result.get_percent_helix()*100 - pred) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {pred}, {pept}, method: 1s"


def test_paper_3_figure_3B():
    """Data from
    https://doi.org/10.1006/jmbi.1994.0024
    
    Only used first data point, triangles.
    The values are approximate, expect some error.
    """
    tol = 3.0
    pept = 'KDESYEELLRKTKAELLHWTKELTEEEKKALAEEGKITB'
    pred = 19.9
    model = AGADIR(method='1s', T=2.0)
    result = model.predict(pept)
    assert abs(result.get_percent_helix()*100 - pred) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {pred}, {pept}, method: 1s"


def test_paper_3_figure_3C():
    """Data from
    https://doi.org/10.1006/jmbi.1994.0024
    
    Only used first data point, triangles.
    The values are approximate, expect some error.
    """
    tol = 3.0
    pept = 'EDAVEALRKLQAGGY'
    pred = 39.0
    model = AGADIR(method='1s', T=5.0)
    result = model.predict(pept)
    assert abs(result.get_percent_helix()*100 - pred) < tol, f"predicted: {round(result.get_percent_helix()*100, 1)}, expected: {pred}, {pept}, method: 1s"
