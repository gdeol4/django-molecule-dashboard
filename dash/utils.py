# dash/utils.py
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64

def mol_to_png(mol):
    pil_img = Draw.MolToImage(mol)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def smiles_to_png(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol_to_png(mol)

# dash/utils.py
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import io

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

def smiles_to_svg(smiles, size=(300, 300)):
    mol = Chem.MolFromSmiles(smiles)
    
    # Create a drawer object
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    
    # Customize the color palette with natural tones
    drawer.drawOptions().updateAtomPalette({
        6: (0.4, 0.2, 0.1),  # Carbon (dark brown)
        7: (0.2, 0.4, 0.2),  # Nitrogen (green)
        8: (0.4, 0.6, 0.2),  # Oxygen (olive green)
        16: (0.6, 0.4, 0.2), # Sulfur (brown)
    })
    
    # Adjust bond line width and font size
    drawer.SetLineWidth(3)  # Increase bond line width
    drawer.SetFontSize(1.2)  # Adjust font size
    
    # Apply ACS style
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg

# dash/utils.py
from .models import GeneratedFlavonoid, AdmetProperties, ProteinTargetPrediction, SuperPredTargetPrediction, SuperPredIndication
import pandas as pd

def get_flavonoid_data():
    flavonoid = GeneratedFlavonoid.objects.first()

    if flavonoid:
        admet_properties = AdmetProperties.objects.filter(inchikey=flavonoid).first()
        protein_target_predictions = ProteinTargetPrediction.objects.filter(inchikey=flavonoid)
        superpred_target_predictions_df = get_superpred_target_predictions(flavonoid.inchikey)
        superpred_indications_df = get_superpred_indications(flavonoid.inchikey)

        data = {
            'flavonoid': {
                'inchikey': flavonoid.inchikey,
                'smiles': flavonoid.smiles,
                'molecular_weight': flavonoid.molecular_weight,
                # Add other fields from GeneratedFlavonoid as needed
            },
            'admet_properties': admet_properties,
            'protein_target_predictions': protein_target_predictions,
            'superpred_target_predictions_df': superpred_target_predictions_df.to_html(index=False, classes='table table-striped'),
            'superpred_indications_df': superpred_indications_df.to_html(index=False, classes='table table-striped')
        }
    else:
        data = {}

    return data

def get_superpred_target_predictions(inchikey):
    superpred_target_predictions = SuperPredTargetPrediction.objects.filter(inchikey=inchikey)
    data = []
    for prediction in superpred_target_predictions:
        data.append({
            'target_name': prediction.target_name,
            'id_chembl': prediction.id_chembl,
            'id_uniprot': prediction.id_uniprot,
            'id_pdb': prediction.id_pdb,
            'id_tdd': prediction.id_tdd,
            'probability': prediction.probability,
            'model_accuracy': prediction.model_accuracy
        })
    df = pd.DataFrame(data)
    return df

def get_superpred_indications(inchikey):
    superpred_indications = SuperPredIndication.objects.filter(inchikey=inchikey)
    data = []
    for indication in superpred_indications:
        data.append({
            'target_name': indication.target_name,
            'id_chembl': indication.id_chembl,
            'indication': indication.indication,
            'probability': indication.probability,
            'model_accuracy': indication.model_accuracy
        })
    df = pd.DataFrame(data)
    return df