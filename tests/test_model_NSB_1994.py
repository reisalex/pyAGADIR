# pytest module

import numpy as np
from pyagadir.models import AGADIR

def calculate_r_squared(y_true, y_pred):
    """
    Calculate the R² value from scratch.
    
    R² = 1 - (Sum of squared residuals / Total sum of squares)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the mean of the observed data
    y_mean = np.mean(y_true)
    
    # Calculate the total sum of squares
    ss_tot = np.sum((y_true - y_mean)**2)
    
    # Calculate the sum of squared residuals
    ss_res = np.sum((y_true - y_pred)**2)
    
    # Calculate R²
    r_squared = 1 - (ss_res / ss_tot)
    
    return r_squared

def test_peptides_in_NSB1994_paper():
    """Data from Munoz & Serrano Nature Structural & Molecular Biology 1994 paper
    https://www.nature.com/articles/nsb0694-399
    https://doi.org/10.1038/nsb0694-399
    
    Here we evaluate pyAGADIR on this test set from the original paper.
    
    See Table 3.
    """

    method = '1s'
    model  = AGADIR(method=method)

    table_3_peptides = {
        'TYKVTELINEAEGINETIDCDD': 1,
        'GFTNSLRMLQQKRWDEAVNLAKS': 10,
        'GVAGFTNSLRMLQQKRWDEAAVNLAKS': 12,
        'ESLLERITRKLRDGWKRLIDIL': 8,
        'ESLLERITRKL': 15,
        'RDGWKRLIDIL': 4,
        'RITRKLRDGWK': 2,
        'KVATTKAQRKLFFNLRKTKQRL': 9,
        'DHPAVMEGTKTILETDSNLS': 4,
        'EPSEQFIKQHDFSSY': 3,
        'VNGMELSKQILQENPH': 6,
        'EVEDYFEEAIRAGLH': 20,
        'KEKITQYIYHVLNGEIL': 3,
        'AVGKSNLLSRYARNEFSA': 2,
        'RFRAVTSAYYRGAVG': 3,
        'TRRTTFESVGRWLDELKIHSD': 7.5,
        'AVSVEEGKALAEEEGLF': 4,
        'STNVKTAFEMVILDIYNNV': 3,
        'DTYKLILNGKTLKGETTTEA': 2,
        'GDAATAEKVFKKIANDNGVD': 4,
        'GEWTYDDATKTFTVTE': 2,
    }

    measured_values = []
    predicted_values = []

    for peptide, measured_value in table_3_peptides.items():
        result = model.predict(peptide)
        predicted_value = result.get_percent_helix() * 100.0
        
        measured_values.append(measured_value)
        predicted_values.append(predicted_value)

        print(f"Peptide: {peptide}")
        print(f"Measured: {measured_value:.2f}, Predicted: {predicted_value:.2f}")
        print("---")

    r_squared = calculate_r_squared(measured_values, predicted_values)
    print(f"R² value: {r_squared:.4f}")

    r_squared_threshold = 0.4

    assert r_squared > r_squared_threshold, (
        f"R² value ({r_squared:.4f}) is below the threshold ({r_squared_threshold})"
    )

if __name__ == "__main__":
    test_peptides_in_NSB1994_paper()