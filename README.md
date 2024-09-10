# α-helix probability model (AGADIR)

**This repository is a work in progress and currently does not produce correct predictions**

An open-source, Python implementation of Munoz & Serrano's AGADIR model of α-helix formation. This model uses statistical mechanics and energy parameters trained on a database of over 400 peptides to predict the α-helical tendency (probability) per residue for a given peptide (see references).

The energy parameters used in this model were extracted from Munoz, V., & Serrano, L. (1995). https://doi.org/10.1006/jmbi.1994.0023

## Install

Install the computational environment with Conda (https://conda.io).

```bash
conda env create -f environment.yml
```

## Usage

The most simple way to use this package is to import and invoke `predict_alphahelix()` where `result.helical_propensity` is the probability that each residue is the α-helical conformation (list of floats) and `result.percent_helix` is the mean helical propensity (probability) for the full peptide (float):
```python
>>> from pyagadir import predict_alphahelix
>>> result = predict_alphahelix('ILKSLEEFLKVTLRSTRQT')
>>> print(f'Percent helix: {result.percent_helix}')
>>> print(f'Per-residue helical propensity: {result.helical_propensity}')
```
```
Percent helix: 0.092
Per-residue helical propensity: [0.00734307 0.01717528 0.03517554 0.13830898 0.16129371 0.17397703
 0.17788564 0.17859396 0.17903603 0.17499225 0.14250647 0.12157049
 0.10387933 0.07653458 0.02485916 0.01393712 0.00978755 0.00462415
 0.00114698]
```

Advanced users may want to modify the partition function to an alternate approximation (e.g. residue, `'r'`) or inspect the detailed dG predicted values. The model class `AGADIR` can be directly imported and invoked. The result object is an instance of `ModelResult` (found in `pyagadir.models`) with more detailed free energy values saved during calculation (stored values are listed below). Example:
```python
>>> from pyagadir.models import AGADIR
>>> model = AGADIR(method='r')
>>> result = model.predict('ILKSLEEFLKVTLRSTRQT')
>>> print(f'dG_Int array (kcal/mol): {result.int_array}')
```
```
dG_Int array (kcal/mol): [0.96 0.8  0.76 1.13 0.8  0.95 0.95 1.08 0.8  0.76 1.12 1.18 0.8  0.67
 1.13 1.18 0.67 0.93 1.18]
```

## Stored Data in ModelResult

```
> seq       :: peptide sequence (str)

# for each residue/index position
> int_array :: dG_Int   (np.array of shape(seq,1))
> i1_array  :: dG_i,i+1 (np.array of shape(seq,1))
> i3_array  :: dG_i,i+3 (np.array of shape(seq,1))
> i4_array  :: dG_i,i+4 (np.array of shape(seq,1))
> N_array   :: dG_Ncap  (np.array of shape(seq,1))
> C_array   :: dG_Ccap  (np.array of shape(seq,1))

> dG_dict_mat :: dG_dict's in list of lists where indexing corresponds to [j][i] (see Muñoz, V., & Serrano, L. (1994)); dG_dict includes each term used in computing dG_Helix for a given helical segment of length j at position i (Python indexing).

# statistical weights and partition functions
> K_tot       :: sum of statistical weights for AGADIR1s (one-sequence) (float)
> K_tot_array :: array of summed statistical weights for AGADIR (residue) (np.array of shape(seq,1))
> Z           :: residue parition function for AGADIR1s (one-sequence) (float)
> Z_array     :: residue parition function for AGADIR (residue) (np.array of shape(seq,1))

# final predicted values
> helical_propensity :: probability that each residue is in the alpha-helical conformation (np.array of shape(seq,1))
> percent_helix      :: mean helical propensity, or probability of peptide is an alpha-helix (float)
```

## To Do

* Implement multiple-sequence approximation (Munoz, V., & Serrano, L. (1997))
* Cythonize the model
* pytests

## For developers

Build package with build (see https://github.com/pypa/build)
```
python -m build
```

## Citations

Muñoz, V., & Serrano, L. (1994). Elucidating the folding problem of helical peptides using empirical parameters. Nature structural biology, 1(6), 399-409. https://doi.org/10.1038/nsb0694-399

Munoz, V., & Serrano, L. (1995). Elucidating the folding problem of helical peptides using empirical parameters. II†. Helix macrodipole effects and rational modification of the helical content of natural peptides. Journal of molecular biology, 245(3), 275-296. https://doi.org/10.1006/jmbi.1994.0023

Muñoz, V., & Serrano, L. (1995). Elucidating the Folding Problem of Helical Peptides using Empirical Parameters. III> Temperature and pH Dependence. Journal of molecular biology, 245(3), 297-308. https://doi.org/10.1006/jmbi.1994.0024

Lacroix, E., Viguera, A. R., & Serrano, L. (1998). Elucidating the folding problem of α-helices: local motifs, long-range electrostatics, ionic-strength dependence and prediction of NMR parameters. Journal of molecular biology, 284(1), 173-191. https://doi.org/10.1006/jmbi.1998.2145

Munoz, V., & Serrano, L. (1997). Development of the multiple sequence approximation within the AGADIR model of α‐helix formation: Comparison with Zimm‐Bragg and Lifson‐Roig formalisms. Biopolymers: Original Research on Biomolecules, 41(5), 495-509. [https://doi.org/10.1002/(SICI)1097-0282(19970415)41:5<495::AID-BIP2>3.0.CO;2-H](https://doi.org/10.1002/(SICI)1097-0282(19970415)41:5<495::AID-BIP2>3.0.CO;2-H)

