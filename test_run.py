# Test script
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pyagadir.models import AGADIR

parser = argparse.ArgumentParser(description="AGADIR test")

    
parser.add_argument(
    '-p', '--peptide',
    type=str,
    required=False,
    help="Peptide sequence",
    default="AAAAAA"
)

parser.add_argument(
    '-m', '--method',
    type=str,
    required=False,
    help="method",
    default="1s"
)

parser.add_argument(
    '-t', '--temperature',
    type=float,
    required=False,
    help="Temperature in C",
    default=0.0
)

parser.add_argument(
    '-i', '--ionic_strength',
    type=float,
    required=False,
    help="Ionic strength in moles",
    default=0.1
)

parser.add_argument(
    '-pH', '--pH',
    type=float,
    required=False,
    help="pH value",
    default=7.0
)

args = parser.parse_args()

model = AGADIR(method=args.method, T=args.temperature, M=args.ionic_strength, pH=args.pH)
#result = model.predict(args.peptide)
result = model.predict('AAADKAAA')
print(result)


def reproduce_figure_3b():
    """
    Reproduce the figure 3b from the AGADIR paper.
    """
    # extracted from figure 3b of the 1998 lacroix paper
    paper_measured_data_ph = [3.02, 3.28, 
                                3.48, 4.08, 4.28, 4.76, 
                                5.07, 5.47, 6.01, 6.37, 
                                6.87, 6.97, 7.27, 8.63, 
                                9.1, 9.37, 9.74, 9.94, 
                                10.18, 10.44, 10.67, 10.87, 
                                11.06, 11.32, 11.8, 
                                ]
    paper_measured_data_helix = [45.3, 45.3, 
                                45.4, 44.8, 44.9, 45.1, 
                                45.4, 45, 45.9, 45.4, 
                                45.3, 45.5, 45.8, 46, 
                                46.4, 48, 48.1, 48.5, 
                                49.2, 50.1, 50.7, 51.5, 
                                52.2, 52.8, 53.2, 
                                ]
    paper_predicted_data_ph = [2.99, 3.19,
                                3.39, 3.58, 3.77, 3.98,
                                4.17, 4.37, 4.56, 4.76,
                                4.96, 5.16, 5.36, 5.56,
                                5.75, 5.96, 6.16, 6.36,
                                6.56, 6.75, 6.96, 7.16,
                                7.36, 7.57, 7.76, 7.96, 
                                8.17, 8.37, 8.57, 8.78,
                                8.97, 9.18, 9.38, 9.58,
                                9.79, 9.99, 10.19, 10.38,
                                10.59, 10.78, 10.99, 11.17,
                                11.38, 11.59, 11.79, 11.99
                                ]
    paper_predicted_data_helix = [37.7, 37.7, 
                                37.7, 37.7, 37.7, 37.7, 
                                37.7, 37.7, 37.7, 37.7, 
                                37.7, 37.7, 37.7, 37.7, 
                                37.7, 37.7, 37.7, 37.7, 
                                37.7, 37.7, 37.7, 37.7, 
                                37.7, 37.7, 37.8, 37.9, 
                                37.9, 38, 38.2, 38.3, 
                                38.7, 39, 39.5, 40, 
                                40.5, 41, 41.5, 42, 
                                42.4, 42.7, 42.9, 43, 
                                43.1, 43.2, 43.3, 43.4, 
                                ]

    # AGADIR results
    pept = 'ZYGGSAAAAAAAKRAAAB' # KR-1a from the paper
    pyagadir_predicted_data_helix = []
    for ph in paper_predicted_data_ph:
        model = AGADIR(method='1s', T=0.0, M=0.1, pH=ph)
        result = model.predict(pept)
        pyagadir_predicted_data_helix.append(result.get_percent_helix())

    # plot
    fig, ax = plt.subplots()
    # plot measured as black circles
    ax.plot(paper_measured_data_ph, paper_measured_data_helix, 'o', color='black', label='Paper measured')

    # plot predicted as white circles with black border
    ax.plot(paper_predicted_data_ph, paper_predicted_data_helix, 'o', color='white', markeredgecolor='black', markeredgewidth=1, label='Paper predicted')

    # set the x-axis to be the same as the paper
    ax.set_xlim(2.8, 12)
    ax.set_ylim(0, 65)

    # plot pyagadir as orange circles
    ax.plot(paper_predicted_data_ph, pyagadir_predicted_data_helix, 'o', color='orange', label='PyAGADIR')
    ax.set_xlabel('pH')
    ax.set_ylabel('Helix content (%)')
    ax.legend()

    # save
    fig.savefig('figure_3b.png')

reproduce_figure_3b()
