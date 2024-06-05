# dash/views.py
from django.shortcuts import render
from .utils import smiles_to_svg

def index(request):
    smiles = 'CC(=O)Oc1ccccc1C(=O)O'  # Replace with your actual SMILES string
    mol_svg = smiles_to_svg(smiles, size=(400, 400))
    context = {'mol_svg': mol_svg}
    return render(request, 'dash/dashboard.html', context)