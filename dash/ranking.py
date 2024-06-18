# # dash/ranking.py

# import sqlite3
# import pandas as pd
# from django.conf import settings
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem.Descriptors import qed
# from rdkit.Chem.Crippen import MolLogP, MolMR
# from rdkit.Chem.Descriptors import ExactMolWt, BertzCT
# from rdkit.Chem.Lipinski import NumHDonors, NumHAcceptors, NumRotatableBonds
# from rdkit.Chem.rdMolDescriptors import CalcTPSA

# def calculate_sa_score(mol):
#     """Calculate the SA score for a given molecule."""
#     mw = ExactMolWt(mol)
#     logp = MolLogP(mol)
#     mr = MolMR(mol)
#     nha = NumHAcceptors(mol)
#     nhd = NumHDonors(mol)
#     nrb = NumRotatableBonds(mol)
#     tpsa = CalcTPSA(mol)
#     bertz = BertzCT(mol)
    
#     mw_score = 1 - (mw - 200) / 800
#     logp_score = 1 - (logp - 2) / 4
#     mr_score = 1 - (mr - 60) / 110
#     nha_score = 1 - nha / 10
#     nhd_score = 1 - nhd / 5
#     nrb_score = 1 - nrb / 10
#     tpsa_score = 1 - (tpsa - 20) / 120
#     bertz_score = 1 - (bertz - 500) / 1500
    
#     score = (mw_score + logp_score + mr_score + nha_score + nhd_score + nrb_score + tpsa_score + bertz_score) / 8
#     return score

# def calculate_qed_score(mol):
#     """Calculate the QED score for a given molecule."""
#     return qed(mol)

# def normalize_data(df, relevant_columns):
#     """Normalize the data in the DataFrame using MinMaxScaler."""
#     numeric_columns = df[relevant_columns].select_dtypes(include=[np.number]).columns
#     scaler = MinMaxScaler()
#     df_numeric = pd.DataFrame(scaler.fit_transform(df[numeric_columns]), columns=numeric_columns, index=df.index)
#     df_normalized = pd.concat([df_numeric, df[relevant_columns].select_dtypes(exclude=[np.number])], axis=1)
#     return df_normalized

# def calculate_entropy_weights(df, relevant_columns):
#     """Calculate entropy-based weights for the DataFrame columns."""
#     numeric_columns = df[relevant_columns].select_dtypes(include=[np.number]).columns
#     probabilities = df[numeric_columns].div(df[numeric_columns].sum(axis=0), axis=1)
#     probabilities = probabilities.replace(0, np.finfo(float).eps)  # Avoid log(0)
#     entropies = -np.sum(probabilities * np.log(probabilities), axis=0)
#     weights = entropies / entropies.sum()
    
#     # Assign equal weights to non-numeric columns within relevant_columns
#     non_numeric_columns = set(relevant_columns) - set(numeric_columns)
#     for column in non_numeric_columns:
#         weights[column] = 1 / len(non_numeric_columns)
    
#     return weights

# def calculate_penalty(df, affinity_column):
#     """Calculate the penalty based on the specified affinity column."""
#     penalty_factor = 10  # Adjust based on desired strength of penalty and specific project requirements
#     df['weighted_score'] -= penalty_factor * (1 - df['QED']) * (1 - df['SA']) * df[affinity_column]
#     return df

# def calculate_scores_and_rank(df):
#     """Calculate scores for molecules and determine their rank."""
#     numeric_columns = df.select_dtypes(include=[np.number]).columns
#     df['score'] = df[numeric_columns].sum(axis=1)
#     df['rank'] = df['score'].rank(ascending=False)
#     return df

# def calculate_scores(df):
#     """Calculate SA and QED scores for the molecules."""
#     df['SA'] = df['smiles'].apply(lambda x: calculate_sa_score(Chem.MolFromSmiles(x)))
#     df['QED'] = df['smiles'].apply(lambda x: calculate_qed_score(Chem.MolFromSmiles(x)))
#     return df

# def rank_molecules(subset_df, protein_target, properties_better_lower):
#     subset_df_normalized = subset_df.iloc[:, 2:].copy()
    
#     # Invert the values of properties_better_lower that are present in the DataFrame
#     properties_to_invert = [prop for prop in properties_better_lower if prop in subset_df_normalized.columns]
#     subset_df_normalized[properties_to_invert] = 1 - subset_df_normalized[properties_to_invert]
    
#     numeric_columns = subset_df_normalized.select_dtypes(include=[np.number]).columns
#     affinity_columns = [f'{protein_target}_neg_log10_affinity_M', f'{protein_target}_affinity_uM']
#     relevant_columns = numeric_columns.union(affinity_columns)
    
#     subset_df_normalized = normalize_data(subset_df_normalized, relevant_columns)
#     weights = calculate_entropy_weights(subset_df_normalized, relevant_columns)
    
#     subset_df_normalized['weighted_score'] = (subset_df_normalized[relevant_columns] * weights[relevant_columns]).sum(axis=1)
    
#     subset_df_normalized = calculate_penalty(subset_df_normalized, f'{protein_target}_affinity_uM')
#     rank_df = calculate_scores_and_rank(subset_df_normalized)
    
#     return rank_df

# def prepare_final_dataframe(df, rank_df, subset_df, protein_target):
#     """Prepare the final DataFrame by combining the original DataFrame with the rank and score columns."""
#     final_columns = ['inchikey', 'smiles'] + [col for col in df.columns if col not in ['inchikey', 'smiles']]
#     final_df = df[final_columns]
#     final_df = pd.concat([final_df, rank_df[['rank', 'score']]], axis=1)
#     final_df = final_df[['inchikey', 'smiles'] + [col for col in final_df.columns if col.startswith(protein_target) or col in ['rank', 'score']]]
#     final_df = pd.concat([final_df, subset_df[[f'{protein_target}_affinity_uM']]], axis=1)
#     final_df.sort_values(by=['rank'], inplace=True)
#     return final_df

# def process_molecules(df, protein_target, properties_better_lower):
#     """Process molecules by calculating scores, ranking, and preparing the final DataFrame."""
#     df = calculate_scores(df)
#     # Create a copy of the DataFrame to be used as subset_df
#     subset_df = df.copy()
#     rank_df = rank_molecules(subset_df, protein_target, properties_better_lower)
#     final_df = prepare_final_dataframe(df, rank_df, subset_df, protein_target)
#     return final_df

# properties_better_higher = [
#     'caco2',
#     'hia',
#     'solubility',
#     'clearance_hepatocyte',
#     'clearance_microsome',
#     'bioavailability',
#     'half_life',
#     'ld50',
#     'alox5_neg_log10_affinity_M',
#     'casp1_neg_log10_affinity_M',
#     'cox1_neg_log10_affinity_M',
#     'flap_neg_log10_affinity_M',
#     'jak1_neg_log10_affinity_M',
#     'jak2_neg_log10_affinity_M',
#     'lck_neg_log10_affinity_M',
#     'magl_neg_log10_affinity_M',
#     'mpges1_neg_log10_affinity_M',
#     'pdl1_neg_log10_affinity_M',
#     'trka_neg_log10_affinity_M',
#     'trkb_neg_log10_affinity_M',
#     'tyk2_neg_log10_affinity_M',
#     'qed'
# ]

# properties_better_lower = [
#     'pgp',
#     'bbb',
#     'ppbr',
#     'vdss',
#     'cyp2c9',
#     'cyp2d6',
#     'cyp3a4',
#     'cyp2c9_substrate',
#     'cyp2d6_substrate',
#     'cyp3a4_substrate',
#     'herg',
#     'ames',
#     'dili',
#     'molecular_weight',
#     'nhet',
#     'nrot',
#     'nring',
#     'nha',
#     'nhd',
#     'logp',
#     'SA',
#     'alox5_affinity_uM',
#     'casp1_affinity_uM',
#     'cox1_affinity_uM',
#     'flap_affinity_uM',
#     'jak1_affinity_uM',
#     'jak2_affinity_uM',
#     'lck_affinity_uM',
#     'magl_affinity_uM',
#     'mpges1_affinity_uM',
#     'pdl1_affinity_uM',
#     'trka_affinity_uM',
#     'trkb_affinity_uM',
#     'tyk2_affinity_uM'
# ]

# excluded_properties = ['inchikey', 'smiles', 'lipophilicity']

# def rank_molecules_for_target(protein_target):
#     # Connect to the SQLite database
#     conn = sqlite3.connect(settings.DATABASES['default']['NAME'])

#     # Define the list of fields to fetch
#     fields = [
#         "inchikey",
#         "ames",
#         "bbb",
#         "bioavailability",
#         "caco2",
#         "clearance_hepatocyte",
#         "clearance_microsome",
#         "cyp2c9",
#         "cyp2c9_substrate",
#         "cyp2d6",
#         "cyp2d6_substrate",
#         "cyp3a4",
#         "cyp3a4_substrate",
#         "dili",
#         "half_life",
#         "herg",
#         "hia",
#         "ld50",
#         "lipophilicity",
#         "pgp",
#         "ppbr",
#         "solubility",
#         "vdss",
#         "alox5_neg_log10_affinity_M",
#         "alox5_affinity_uM",
#         "molecular_weight",
#         "nhet",
#         "nrot",
#         "nring",
#         "nha",
#         "nhd",
#         "logp",
#         "smiles",
#         "casp1_neg_log10_affinity_M",
#         "casp1_affinity_uM",
#         "cox1_neg_log10_affinity_M",
#         "cox1_affinity_uM",
#         "flap_neg_log10_affinity_M",
#         "flap_affinity_uM",
#         "jak1_neg_log10_affinity_M",
#         "jak1_affinity_uM",
#         "jak2_neg_log10_affinity_M",
#         "jak2_affinity_uM",
#         "lck_neg_log10_affinity_M",
#         "lck_affinity_uM",
#         "magl_neg_log10_affinity_M",
#         "magl_affinity_uM",
#         "mpges1_neg_log10_affinity_M",
#         "mpges1_affinity_uM",
#         "pdl1_neg_log10_affinity_M",
#         "pdl1_affinity_uM",
#         "trka_neg_log10_affinity_M",
#         "trka_affinity_uM",
#         "trkb_neg_log10_affinity_M",
#         "trkb_affinity_uM",
#         "tyk2_neg_log10_affinity_M",
#         "tyk2_affinity_uM"
#     ]

#     # Construct the SQL query to fetch the specified fields
#     query = f"""
#     SELECT
#         gf.inchikey,
#         ap.ames,
#         ap.bbb,
#         ap.bioavailability,
#         ap.caco2,
#         ap.clearance_hepatocyte,
#         ap.clearance_microsome,
#         ap.cyp2c9,
#         ap.cyp2c9_substrate,
#         ap.cyp2d6,
#         ap.cyp2d6_substrate,
#         ap.cyp3a4,
#         ap.cyp3a4_substrate,
#         ap.dili,
#         ap.half_life,
#         ap.herg,
#         ap.hia,
#         ap.ld50,
#         ap.lipophilicity,
#         ap.pgp,
#         ap.ppbr,
#         ap.solubility,
#         ap.vdss,
#         ptp.alox5_neg_log10_affinity_M,
#         ptp.alox5_affinity_uM,
#         gf.molecular_weight,
#         gf.nhet,
#         gf.nrot,
#         gf.nring,
#         gf.nha,
#         gf.nhd,
#         gf.logp,
#         gf.smiles,
#         ptp.casp1_neg_log10_affinity_M,
#         ptp.casp1_affinity_uM,
#         ptp.cox1_neg_log10_affinity_M,
#         ptp.cox1_affinity_uM,
#         ptp.flap_neg_log10_affinity_M,
#         ptp.flap_affinity_uM,
#         ptp.jak1_neg_log10_affinity_M,
#         ptp.jak1_affinity_uM,
#         ptp.jak2_neg_log10_affinity_M,
#         ptp.jak2_affinity_uM,
#         ptp.lck_neg_log10_affinity_M,
#         ptp.lck_affinity_uM,
#         ptp.magl_neg_log10_affinity_M,
#         ptp.magl_affinity_uM,
#         ptp.mpges1_neg_log10_affinity_M,
#         ptp.mpges1_affinity_uM,
#         ptp.pdl1_neg_log10_affinity_M,
#         ptp.pdl1_affinity_uM,
#         ptp.trka_neg_log10_affinity_M,
#         ptp.trka_affinity_uM,
#         ptp.trkb_neg_log10_affinity_M,
#         ptp.trkb_affinity_uM,
#         ptp.tyk2_neg_log10_affinity_M,
#         ptp.tyk2_affinity_uM
#     FROM
#         generated_flavonoids gf
#         JOIN admet_properties ap ON gf.inchikey = ap.inchikey_id
#         JOIN protein_target_predictions ptp ON gf.inchikey = ptp.inchikey_id
#     """

# dash/ranking.py

# import pandas as pd

# from django.conf import settings

# from sklearn.preprocessing import MinMaxScaler

# import numpy as np

# from rdkit import Chem

# from rdkit.Chem.Descriptors import qed

# from rdkit.Chem.Crippen import MolLogP, MolMR

# from rdkit.Chem.Descriptors import ExactMolWt, BertzCT

# from rdkit.Chem.Lipinski import NumHDonors, NumHAcceptors, NumRotatableBonds

# from rdkit.Chem.rdMolDescriptors import CalcTPSA

# from .models import GeneratedFlavonoid, AdmetProperties, ProteinTargetPrediction

# def calculate_sa_score(mol):
#     """Calculate the SA score for a given molecule."""
#     mw = ExactMolWt(mol)
#     logp = MolLogP(mol)
#     mr = MolMR(mol)
#     nha = NumHAcceptors(mol)
#     nhd = NumHDonors(mol)
#     nrb = NumRotatableBonds(mol)
#     tpsa = CalcTPSA(mol)
#     bertz = BertzCT(mol)

#     mw_score = 1 - (mw - 200) / 800
#     logp_score = 1 - (logp - 2) / 4
#     mr_score = 1 - (mr - 60) / 110
#     nha_score = 1 - nha / 10
#     nhd_score = 1 - nhd / 5
#     nrb_score = 1 - nrb / 10
#     tpsa_score = 1 - (tpsa - 20) / 120
#     bertz_score = 1 - (bertz - 500) / 1500

#     score = (mw_score + logp_score + mr_score + nha_score + nhd_score + nrb_score + tpsa_score + bertz_score) / 8
#     return score

# def calculate_qed_score(mol):
#     """Calculate the QED score for a given molecule."""
#     return qed(mol)

# def normalize_data(df, relevant_columns):
#     """Normalize the data in the DataFrame using MinMaxScaler."""
#     numeric_columns = df[relevant_columns].select_dtypes(include=[np.number]).columns
#     scaler = MinMaxScaler()
#     df_numeric = pd.DataFrame(scaler.fit_transform(df[numeric_columns]), columns=numeric_columns, index=df.index)
#     df_normalized = pd.concat([df_numeric, df[relevant_columns].select_dtypes(exclude=[np.number])], axis=1)
#     return df_normalized

# def calculate_entropy_weights(df, relevant_columns):
#     """Calculate entropy-based weights for the DataFrame columns."""
#     numeric_columns = df[relevant_columns].select_dtypes(include=[np.number]).columns
#     probabilities = df[numeric_columns].div(df[numeric_columns].sum(axis=0), axis=1)
#     probabilities = probabilities.replace(0, np.finfo(float).eps)  # Avoid log(0)
#     entropies = -np.sum(probabilities * np.log(probabilities), axis=0)
#     weights = entropies / entropies.sum()

#     # Assign equal weights to non-numeric columns within relevant_columns
#     non_numeric_columns = set(relevant_columns) - set(numeric_columns)
#     for column in non_numeric_columns:
#         weights[column] = 1 / len(non_numeric_columns)

#     return weights

# def calculate_penalty(df, affinity_column):
#     """Calculate the penalty based on the specified affinity column."""
#     penalty_factor = 10  # Adjust based on desired strength of penalty and specific project requirements
#     df['weighted_score'] -= penalty_factor * (1 - df['QED']) * (1 - df['SA']) * df[affinity_column]
#     return df

# def calculate_scores_and_rank(df):
#     """Calculate scores for molecules and determine their rank."""
#     numeric_columns = df.select_dtypes(include=[np.number]).columns
#     df['score'] = df[numeric_columns].sum(axis=1)
#     df['rank'] = df['score'].rank(ascending=False)
#     return df

# def calculate_scores(df):
#     """Calculate SA and QED scores for the molecules."""
#     df['SA'] = df['smiles'].apply(lambda x: calculate_sa_score(Chem.MolFromSmiles(x)))
#     df['QED'] = df['smiles'].apply(lambda x: calculate_qed_score(Chem.MolFromSmiles(x)))
#     return df

# def rank_molecules(subset_df, protein_target, properties_better_lower):
#     subset_df_normalized = subset_df.iloc[:, 2:].copy()

#     # Invert the values of properties_better_lower that are present in the DataFrame
#     properties_to_invert = [prop for prop in properties_better_lower if prop in subset_df_normalized.columns]
#     subset_df_normalized[properties_to_invert] = 1 - subset_df_normalized[properties_to_invert]

#     numeric_columns = subset_df_normalized.select_dtypes(include=[np.number]).columns
#     affinity_columns = [f'{protein_target}_neg_log10_affinity_M', f'{protein_target}_affinity_uM']
#     relevant_columns = numeric_columns.union(affinity_columns)

#     subset_df_normalized = normalize_data(subset_df_normalized, relevant_columns)
#     weights = calculate_entropy_weights(subset_df_normalized, relevant_columns)
#     subset_df_normalized['weighted_score'] = (subset_df_normalized[relevant_columns] * weights[relevant_columns]).sum(axis=1)
#     subset_df_normalized = calculate_penalty(subset_df_normalized, f'{protein_target}_affinity_uM')
#     rank_df = calculate_scores_and_rank(subset_df_normalized)

#     return rank_df

# def prepare_final_dataframe(df, rank_df, subset_df, protein_target):
#     """Prepare the final DataFrame by combining the original DataFrame with the rank and score columns."""
#     final_columns = ['inchikey', 'smiles'] + [col for col in df.columns if col not in ['inchikey', 'smiles']]
#     final_df = df[final_columns]
#     final_df = pd.concat([final_df, rank_df[['rank', 'score']]], axis=1)
#     final_df = final_df[['inchikey', 'smiles'] + [col for col in final_df.columns if col.startswith(protein_target) or col in ['rank', 'score']]]
#     final_df = pd.concat([final_df, subset_df[[f'{protein_target}_affinity_uM']]], axis=1)
#     final_df.sort_values(by=['rank'], inplace=True)
#     return final_df

# def process_molecules(df, protein_target, properties_better_lower):
#     """Process molecules by calculating scores, ranking, and preparing the final DataFrame."""
#     df = calculate_scores(df)
#     # Create a copy of the DataFrame to be used as subset_df
#     subset_df = df.copy()
#     rank_df = rank_molecules(subset_df, protein_target, properties_better_lower)
#     final_df = prepare_final_dataframe(df, rank_df, subset_df, protein_target)
#     return final_df

# properties_better_higher = [
#     'caco2',
#     'hia',
#     'solubility',
#     'clearance_hepatocyte',
#     'clearance_microsome',
#     'bioavailability',
#     'half_life',
#     'ld50',
#     'alox5_neg_log10_affinity_M',
#     'casp1_neg_log10_affinity_M',
#     'cox1_neg_log10_affinity_M',
#     'flap_neg_log10_affinity_M',
#     'jak1_neg_log10_affinity_M',
#     'jak2_neg_log10_affinity_M',
#     'lck_neg_log10_affinity_M',
#     'magl_neg_log10_affinity_M',
#     'mpges1_neg_log10_affinity_M',
#     'pdl1_neg_log10_affinity_M',
#     'trka_neg_log10_affinity_M',
#     'trkb_neg_log10_affinity_M',
#     'tyk2_neg_log10_affinity_M',
#     'qed'
# ]

# properties_better_lower = [
#     'pgp',
#     'bbb',
#     'ppbr',
#     'vdss',
#     'cyp2c9',
#     'cyp2d6',
#     'cyp3a4',
#     'cyp2c9_substrate',
#     'cyp2d6_substrate',
#     'cyp3a4_substrate',
#     'herg',
#     'ames',
#     'dili',
#     'molecular_weight',
#     'nhet',
#     'nrot',
#     'nring',
#     'nha',
#     'nhd',
#     'logp',
#     'SA',
#     'alox5_affinity_uM',
#     'casp1_affinity_uM',
#     'cox1_affinity_uM',
#     'flap_affinity_uM',
#     'jak1_affinity_uM',
#     'jak2_affinity_uM',
#     'lck_affinity_uM',
#     'magl_affinity_uM',
#     'mpges1_affinity_uM',
#     'pdl1_affinity_uM',
#     'trka_affinity_uM',
#     'trkb_affinity_uM',
#     'tyk2_affinity_uM'
# ]

# excluded_properties = ['inchikey', 'smiles', 'lipophilicity']

# def rank_molecules_for_target(protein_target):
#     # Fetch the data from the database using Django models
#     generated_flavonoids = GeneratedFlavonoid.objects.all()
#     admet_properties = AdmetProperties.objects.all()
#     protein_target_predictions = ProteinTargetPrediction.objects.all()

#     # Create DataFrames from the fetched data
#     df = pd.DataFrame(list(generated_flavonoids.values()))
#     print("Generated Flavonoids DataFrame:")
#     print(df.head())
#     print(df.columns)

#     admet_properties_df = pd.DataFrame(list(admet_properties.values()))
#     admet_properties_df = admet_properties_df.rename(columns={'inchikey_id': 'inchikey'})
#     print("ADMET Properties DataFrame:")
#     print(admet_properties_df.head())
#     print(admet_properties_df.columns)
    
#     # Merge the DataFrames
#     df = df.merge(admet_properties_df, on='inchikey', how='left')
#     print("Merged DataFrame (Generated Flavonoids + ADMET Properties):")
#     print(df.head())
#     print(df.columns)
    
#     # Rename the 'inchikey_id' column to 'inchikey' in the protein_target_predictions DataFrame
#     protein_target_predictions_df = pd.DataFrame(list(protein_target_predictions.values()))
#     protein_target_predictions_df = protein_target_predictions_df.rename(columns={'inchikey_id': 'inchikey'})
#     print("Protein Target Predictions DataFrame:")
#     print(protein_target_predictions_df.head())
#     print(protein_target_predictions_df.columns)
    
#     df = df.merge(protein_target_predictions_df, on='inchikey', how='left')
#     print("Merged DataFrame (Generated Flavonoids + ADMET Properties + Protein Target Predictions):")
#     print(df.head())
#     print(df.columns)

#     # Check if the 'smiles' column exists in the DataFrame
#     if 'smiles' not in df.columns:
#         print("Error: 'smiles' column not found in the DataFrame.")
#         print("Available columns:", df.columns)
#         return None

#     # Process the molecules and return the final DataFrame
#     final_df = process_molecules(df, protein_target, properties_better_lower)
#     return final_df

import pandas as pd
from .models import GeneratedFlavonoid, AdmetProperties, ProteinTargetPrediction, SuperPredTargetPrediction, SuperPredIndication
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt, BertzCT
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem.Lipinski import NumHDonors, NumHAcceptors, NumRotatableBonds
from rdkit.Chem.rdMolDescriptors import CalcTPSA

# ranking.py
import pandas as pd
from .models import GeneratedFlavonoid, AdmetProperties, ProteinTargetPrediction, SuperPredTargetPrediction, SuperPredIndication
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt, BertzCT
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem.Lipinski import NumHDonors, NumHAcceptors, NumRotatableBonds
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.QED import qed
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def calculate_sa_score(mol):
    """Calculate the SA score for a given molecule."""
    if mol is None:
        return None
    
    mw = ExactMolWt(mol)
    logp = MolLogP(mol)
    mr = MolMR(mol)
    nha = NumHAcceptors(mol)
    nhd = NumHDonors(mol)
    nrb = NumRotatableBonds(mol)
    tpsa = CalcTPSA(mol)
    bertz = BertzCT(mol)
    
    mw_score = 1 - (mw - 200) / 800
    logp_score = 1 - (logp - 2) / 4
    mr_score = 1 - (mr - 60) / 110
    nha_score = 1 - nha / 10
    nhd_score = 1 - nhd / 5
    nrb_score = 1 - nrb / 10
    tpsa_score = 1 - (tpsa - 20) / 120
    bertz_score = 1 - (bertz - 500) / 1500
    
    score = (mw_score + logp_score + mr_score + nha_score + nhd_score + nrb_score + tpsa_score + bertz_score) / 8
    return score

def calculate_qed_score(mol):
    """Calculate the QED score for a given molecule."""
    return qed(mol)

def normalize_data(df, columns):
    """Normalize the specified columns in the DataFrame using MinMaxScaler."""
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def calculate_entropy_weights(df, columns):
    """Calculate entropy-based weights for the specified DataFrame columns."""
    probabilities = df[columns].div(df[columns].sum(axis=0), axis=1)
    probabilities = probabilities.replace(0, np.finfo(float).eps)  # Avoid log(0)
    entropies = -np.sum(probabilities * np.log(probabilities), axis=0)
    weights = entropies / entropies.sum()
    return weights

def calculate_penalty(df, affinity_column):
    """Calculate the penalty based on the specified affinity column."""
    penalty_factor = 10  # Adjust based on desired strength of penalty and specific project requirements
    df['weighted_score'] -= penalty_factor * (1 - df['qed_score']) * (1 - df['sa_score']) * df[affinity_column]
    return df

def calculate_scores_and_rank(df):
    """Calculate scores for molecules and determine their rank."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df['score'] = df[numeric_columns].sum(axis=1)
    df['rank'] = df['score'].rank(ascending=False)
    return df

def rank_molecules(df, protein_target, properties_better_lower):
    subset_df_normalized = df.iloc[:, 2:].copy()
    
    # Invert the values of properties_better_lower that are present in the DataFrame
    properties_to_invert = [prop for prop in properties_better_lower if prop in subset_df_normalized.columns]
    subset_df_normalized[properties_to_invert] = 1 - subset_df_normalized[properties_to_invert]
    
    numeric_columns = subset_df_normalized.select_dtypes(include=[np.number]).columns
    affinity_columns = [f'{protein_target}_neg_log10_affinity_M', f'{protein_target}_affinity_uM']
    relevant_columns = numeric_columns.union(affinity_columns)
    
    subset_df_normalized = normalize_data(subset_df_normalized, relevant_columns)
    weights = calculate_entropy_weights(subset_df_normalized, relevant_columns)
    
    subset_df_normalized['weighted_score'] = (subset_df_normalized[relevant_columns] * weights[relevant_columns]).sum(axis=1)
    
    subset_df_normalized = calculate_penalty(subset_df_normalized, f'{protein_target}_affinity_uM')
    rank_df = calculate_scores_and_rank(subset_df_normalized)
    
    return rank_df

def prepare_final_dataframe(df, rank_df, subset_df, protein_target):
    """Prepare the final DataFrame by combining the original DataFrame with the rank and score columns."""
    final_columns = ['inchikey', 'smiles'] + [col for col in df.columns if col not in ['inchikey', 'smiles']]
    final_df = df[final_columns]
    final_df = pd.concat([final_df, rank_df[['rank', 'score']]], axis=1)
    final_df = final_df[['inchikey', 'smiles'] + [col for col in final_df.columns if col.startswith(protein_target) or col in ['rank', 'score']]]
    final_df = pd.concat([final_df, subset_df[[f'{protein_target}_affinity_uM']]], axis=1)
    final_df.sort_values(by=['rank'], inplace=True)
    return final_df

def get_ranking_data(protein=''):
    flavonoids = GeneratedFlavonoid.objects.all().values('inchikey', 'molecular_weight', 'nhet', 'nrot', 'nring', 'nha', 'nhd', 'logp', 'smiles')
    admet = AdmetProperties.objects.all().values()
    protein_targets = ProteinTargetPrediction.objects.all().values('inchikey', 'alox5_neg_log10_affinity_M', 'alox5_affinity_uM', 'casp1_neg_log10_affinity_M', 'casp1_affinity_uM', 'cox1_neg_log10_affinity_M', 'cox1_affinity_uM', 'flap_neg_log10_affinity_M', 'flap_affinity_uM', 'jak1_neg_log10_affinity_M', 'jak1_affinity_uM', 'jak2_neg_log10_affinity_M', 'jak2_affinity_uM', 'lck_neg_log10_affinity_M', 'lck_affinity_uM', 'magl_neg_log10_affinity_M', 'magl_affinity_uM', 'mpges1_neg_log10_affinity_M', 'mpges1_affinity_uM', 'pdl1_neg_log10_affinity_M', 'pdl1_affinity_uM', 'trka_neg_log10_affinity_M', 'trka_affinity_uM', 'trkb_neg_log10_affinity_M', 'trkb_affinity_uM', 'tyk2_neg_log10_affinity_M', 'tyk2_affinity_uM')

    df_flavonoids = pd.DataFrame(flavonoids)
    df_admet = pd.DataFrame(admet)
    df_protein_targets = pd.DataFrame(protein_targets)

    df = pd.merge(df_flavonoids, df_admet, left_on='inchikey', right_on='inchikey_id')
    df = pd.merge(df, df_protein_targets, on='inchikey')

    # Remove rows with None values in the inchikey column
    df = df.dropna(subset=['inchikey'])
    
    # Create a 'mol' column from the 'smiles' column
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    
    # Calculate SA score for each molecule using the 'mol' column
    df['sa_score'] = df['mol'].apply(calculate_sa_score)
    
    # Calculate QED score for each molecule using the 'mol' column
    df['qed_score'] = df['mol'].apply(calculate_qed_score)
    
    # Move the 'qed_score' and 'sa_score' columns to the front of the DataFrame
    cols = df.columns.tolist()
    cols = ['qed_score', 'sa_score'] + cols[:-2]
    df = df[cols]
    
    properties_better_lower = [
        'pgp',
        'bbb',
        'ppbr',
        'vdss',
        'cyp2c9',
        'cyp2d6',
        'cyp3a4',
        'cyp2c9_substrate',
        'cyp2d6_substrate',
        'cyp3a4_substrate',
        'herg',
        'ames',
        'dili',
        'molecular_weight',
        'nhet',
        'nrot',
        'nring',
        'nha',
        'nhd',
        'logp',
        'alox5_affinity_uM',
        'casp1_affinity_uM',
        'cox1_affinity_uM',
        'flap_affinity_uM',
        'jak1_affinity_uM',
        'jak2_affinity_uM',
        'lck_affinity_uM',
        'magl_affinity_uM',
        'mpges1_affinity_uM',
        'pdl1_affinity_uM',
        'trka_affinity_uM',
        'trkb_affinity_uM',
        'tyk2_affinity_uM'
    ]
    
    if protein:
        final_df = prepare_final_dataframe(df, rank_molecules(df, protein, properties_better_lower), df, protein)
    else:
        final_df = df
    
    return final_df