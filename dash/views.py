# dash/views.py
from django.shortcuts import render

from dash.models import GeneratedFlavonoid
from .utils import smiles_to_svg

# def index(request):
#     smiles = 'CC(=O)Oc1ccccc1C(=O)O'  # Replace with your actual SMILES string
#     mol_svg = smiles_to_svg(smiles, size=(400, 400))
#     context = {'mol_svg': mol_svg}
#     return render(request, 'dash/dashboard.html', context)

# dash/views.py
from django.shortcuts import render
from .utils import smiles_to_svg, get_flavonoid_data
from django.core.paginator import Paginator

def flavonoid_data_generator():
    flavonoids = GeneratedFlavonoid.objects.all()
    for flavonoid in flavonoids:
        yield get_flavonoid_data(flavonoid.inchikey)

# dash/views.py
from django.shortcuts import render
from .utils import smiles_to_svg, get_flavonoid_data
from django.core.paginator import Paginator
from .models import GeneratedFlavonoid

# dash/views.py
from django.shortcuts import render
from .utils import smiles_to_svg, get_flavonoid_data
from .models import GeneratedFlavonoid

def index(request):
    inchikey = request.GET.get('inchikey')
    inchikeys = GeneratedFlavonoid.objects.values_list('inchikey', flat=True)

    context = {
        'inchikeys': inchikeys,
    }

    if inchikey:
        flavonoid_data = get_flavonoid_data(inchikey)
        smiles = flavonoid_data['flavonoid']['smiles'] if flavonoid_data else ''
        mol_svg = smiles_to_svg(smiles, size=(400, 400))

        if flavonoid_data and 'superpred_indications' in flavonoid_data:
            superpred_indications = flavonoid_data['superpred_indications']
        else:
            superpred_indications = []

        context.update({
            'mol_svg': mol_svg,
            'admet_data': flavonoid_data,
            'superpred_indications': superpred_indications,
        })

    return render(request, 'dash/dashboard.html', context)

# dash/views.py

from django.shortcuts import render
from .ranking import get_ranking_data

def ranking_view(request):
    protein = request.GET.get('protein', '')
    df = None
    if protein:
        df = get_ranking_data(protein)
    return render(request, 'dash/ranking.html', {'df': df, 'protein': protein})

# dash/views.py
from django.shortcuts import render
from .utils import get_protein_targets, get_top_molecules

def rank(request):
    protein_targets = get_protein_targets()
    context = {
        'protein_targets': protein_targets
    }
    return render(request, 'dash/rank.html', context)

def protein_target_detail(request, protein_target):
    top_molecules = get_top_molecules(protein_target)
    context = {
        'protein_target': protein_target,
        'top_molecules': top_molecules
    }
    return render(request, 'dash/protein_target_detail.html', context)

# dash/views.py
from django.shortcuts import render
from .utils import smiles_to_svg, get_flavonoid_data, get_top_molecules

def molecule_dashboard(request, inchikey):
    flavonoid_data = get_flavonoid_data(inchikey)
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