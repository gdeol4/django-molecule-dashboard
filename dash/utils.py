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

def smiles_to_svg(smiles, size=(500, 500)):
    mol = Chem.MolFromSmiles(smiles)
    
    # Create a drawer object
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    
    # Customize the color palette with natural tones
    drawer.drawOptions().updateAtomPalette({
        6: (0.4, 0.2, 0.1),  # Carbon (dark brown)
        7: (0.2, 0.4, 0.2),  # Nitrogen (green)
        8: (0.4, 0.6, 0.2),  # Oxygen (olive green)
        16: (0.6, 0.4, 0.2), # Sulfur (brown)
    })
    
    # Adjust bond line width and font size
    drawer.SetLineWidth(5)  # Increase bond line width
    drawer.SetFontSize(0.8)  # Adjust font size
    
    # Apply ACS style
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg