# α-helix probability model implemented in Python (pyAGADIR)

**This repository is a work in progress and currently does not produce correct predictions**

An open-source, Python implementation of Munoz & Serrano's AGADIR model of α-helix formation. This model uses statistical mechanics and energy parameters trained on a database of over 400 peptides to predict the α-helical tendency (probability) per residue for a given peptide (see references).

The energy parameters used in this model were extracted from the supplementary material of Lacroix, E., Viguera, A. R., & Serrano, L. (1998). Elucidating the folding problem of α-helices: local motifs, long-range electrostatics, ionic-strength dependence and prediction of NMR parameters. Journal of molecular biology, 284(1), 173-191. https://doi.org/10.1006/jmbi.1998.2145

The paper uses the terminology of Richardson & Richardson (1988) where STC (S, strand; T, turn; and C, coil) indicates a non-helical conformation and He is a helical residue. Python indexing starting from the Ncap is used to describe these positions in the model.
```text
Name:      N''  N'   Ncap N1   N2   N3   N4   N5.............C5   C4   C3   C2   C1   Ccap C'   C''  
Structure: STC  STC  STC -He---He---He---He---He---He---He---He---He---He---He---He---STC  STC  STC
Index:     -2   -1   0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15
```


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

## Capping
In many helix stability studies, it is common to "cap" the helix by adding on acylation, succinylation, or amidation at the terminal ends of the helix. These modifications are used to stabilize the helix by neutralizing the charges at the N-terminus or C-terminus, which can otherwise destabilize the helix by introducing repulsive interactions or by interfering with the regular hydrogen bonding pattern that defines the helical structure.

    Acylation is the addition of an acyl group (such as an acetyl group) to the N-terminus. This modification similarly neutralizes the positive charge on the N-terminus, preventing electrostatic interactions that could destabilize the helical structure.

    Succinylation, which involves the attachment of a succinyl group to the N-terminus, can also be used as a modification to modulate charge and influence solubility, often employed for similar stabilizing effects on the helix termini.

    Amidation typically occurs at the C-terminus, where the carboxyl group is converted to an amide, reducing the net negative charge. This can help improve the helix's stability, particularly in peptides, by minimizing the disruption caused by terminal charges.

In pyAgadir we have chosen to represent acylation with the single letter **Z**, succinylation with **X**, and amidation with **B**. The peptide **ILKSLEEFLKVTLRSTRQT**, when capped with acylation and amidation, would be written **ZILKSLEEFLKVTLRSTRQTB** when submitted to pyAgadir. When extracting a single alpha helix from a longer protein chain, such as a pdb structure, you should consider capping it in order to simulate the absence of N- and C-terminal charges and obtain accurate estimates of helicity.

## Questions / To Do

* How is the i+1 term supposed to be calculated?
* How do we add charge interactions in the i+3 and i+4 terms? Potental duplication of electrostatics term?
* Test correct functioning of staple term or schellman term.
* We need to locate a source for the N- and C-terminal pKa values for the individual amino acids. Currently using average value from Stryer.
* Ensure that N- and C-terminal capping is dealt with appropriately. Introduce protein "capping" for accurate estimations of helices in proteins?
* Update pytests to fit new model.


## Citations

Muñoz, V., & Serrano, L. (1994). Elucidating the folding problem of helical peptides using empirical parameters. Nature structural biology, 1(6), 399-409. https://doi.org/10.1038/nsb0694-399

Munoz, V., & Serrano, L. (1995). Elucidating the folding problem of helical peptides using empirical parameters. II†. Helix macrodipole effects and rational modification of the helical content of natural peptides. Journal of molecular biology, 245(3), 275-296. https://doi.org/10.1006/jmbi.1994.0023

Muñoz, V., & Serrano, L. (1995). Elucidating the Folding Problem of Helical Peptides using Empirical Parameters. III> Temperature and pH Dependence. Journal of molecular biology, 245(3), 297-308. https://doi.org/10.1006/jmbi.1994.0024

Lacroix, E., Viguera, A. R., & Serrano, L. (1998). Elucidating the folding problem of α-helices: local motifs, long-range electrostatics, ionic-strength dependence and prediction of NMR parameters. Journal of molecular biology, 284(1), 173-191. https://doi.org/10.1006/jmbi.1998.2145

Munoz, V., & Serrano, L. (1997). Development of the multiple sequence approximation within the AGADIR model of α‐helix formation: Comparison with Zimm‐Bragg and Lifson‐Roig formalisms. Biopolymers: Original Research on Biomolecules, 41(5), 495-509. [https://doi.org/10.1002/(SICI)1097-0282(19970415)41:5<495::AID-BIP2>3.0.CO;2-H](https://doi.org/10.1002/(SICI)1097-0282(19970415)41:5<495::AID-BIP2>3.0.CO;2-H)

