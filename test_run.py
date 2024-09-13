# Test script
import argparse

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
result = model.predict(args.peptide)
print(result)