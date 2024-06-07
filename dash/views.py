# dash/views.py
from django.shortcuts import render
from .utils import smiles_to_svg

# def index(request):
#     smiles = 'CC(=O)Oc1ccccc1C(=O)O'  # Replace with your actual SMILES string
#     mol_svg = smiles_to_svg(smiles, size=(400, 400))
#     context = {'mol_svg': mol_svg}
#     return render(request, 'dash/dashboard.html', context)

# dash/views.py
from django.shortcuts import render
from .utils import smiles_to_svg, get_flavonoid_data
def index(request):
    flavonoid_data = get_flavonoid_data()
    smiles = flavonoid_data['flavonoid']['smiles'] if flavonoid_data else ''
    mol_svg = smiles_to_svg(smiles, size=(400, 400))

    if flavonoid_data and 'superpred_indications' in flavonoid_data:
        superpred_indications = flavonoid_data['superpred_indications']
    else:
        superpred_indications = []

    context = {
        'mol_svg': mol_svg,
        'admet_data': flavonoid_data,
        'superpred_indications': superpred_indications
    }
    return render(request, 'dash/dashboard.html', context)