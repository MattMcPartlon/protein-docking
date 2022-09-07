import torch

AA_ALPHABET = "ARNDCQEGHILKMFPSTWYV-"
AA_INDEX_MAP = {aa: i for i, aa in enumerate(AA_ALPHABET)}
N_AMINO_ACID_KEYS = 21
BB_ATOMS = ['N', 'CA', 'C', 'O']
BB_ATOM_POSNS = {a: i for i, a in enumerate(BB_ATOMS)}
SC_ATOMS = ['CE3', 'CZ', 'SD', 'CB', 'CD1', 'NH1', 'OG1', 'CE1', 'OE1', 'CZ2', 'OH', 'CG',
            'CZ3', 'NE', 'CH2', 'OD1', 'NH2', 'ND2', 'OG', 'CG2', 'OE2', 'CD2', 'ND1', 'NE2',
            'NZ', 'CD', 'CE2', 'CE', 'OD2', 'SG', 'NE1', 'CG1']
SC_ATOM_POSNS = {a: i for i, a in enumerate(SC_ATOMS)}

AA3LetterCode = ['ALA', 'ARG', 'ASN', 'ASP', 'ASX', 'CYS', 'GLU', 'GLN', 'GLX', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
                 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', "UNK"]
AA1LetterCode = ['A', 'R', 'N', 'D', 'B', 'C', 'E', 'Q', 'Z', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
                 'Y', 'V', "-"]
VALID_AA_3_LETTER = set(AA3LetterCode)
VALID_AA_1_LETTER = set(AA1LetterCode)

ALL_ATOMS = BB_ATOMS + SC_ATOMS
ALL_ATOM_POSNS = {a: i for i, a in enumerate(ALL_ATOMS)}

THREE_TO_ONE = {three: one for three, one in zip(AA3LetterCode, AA1LetterCode)}
ONE_TO_THREE = {one: three for three, one in THREE_TO_ONE.items()}

N_BB_ATOMS = len(BB_ATOMS)
N_SEC_STRUCT_KEYS = 9
SS_KEY_MAP = {'S': 1, 'H': 2, 'T': 3, 'I': 4, 'E': 5, 'G': 6, 'L': 7, 'B': 8, '-': 0}

AA_TO_INDEX = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5,
               'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11,
               'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16,
               'TRP': 17, 'TYR': 18, 'VAL': 19}
AA_TO_INDEX.update({THREE_TO_ONE[aa]: v for aa, v in AA_TO_INDEX.items()})

INDEX_TO_AA_ONE = {AA_TO_INDEX[a]: a for a in AA_TO_INDEX.keys() if len(a) == 1}
INDEX_TO_AA_THREE = {AA_TO_INDEX[a]: a for a in AA_TO_INDEX.keys() if len(a) == 3}
AA_TO_N_SC_ATOMS = {'ALA': 1, 'ARG': 7, 'ASN': 4, 'ASP': 4, 'CYS': 2, 'GLN': 5,
                    'GLU': 5, 'GLY': 0, 'HIS': 6, 'ILE': 4, 'LEU': 4, 'LYS': 5,
                    'MET': 4, 'PHE': 7, 'PRO': 3, 'SER': 2, 'THR': 3,
                    'TRP': 10, 'TYR': 8, 'VAL': 3}
AA_TO_N_SC_ATOMS.update({THREE_TO_ONE[aa]: v for aa, v in AA_TO_N_SC_ATOMS.items()})

AA_TO_DISTAL = {'ARG': 'CG', 'ASN': 'CG', 'ASP': 'CG', 'CYS': 'SG',
                'GLN': 'CG', 'GLU': 'CG', 'HIS': 'CG1', 'ILE': 'CG',
                'LEU': 'CG', 'LYS': 'CD', 'MET': 'CG', 'PHE': 'CG',
                'PRO': 'CG', 'SER': 'OG', 'THR': 'OG1', 'TRP': 'CG',
                'TYR': 'CG', 'VAL': 'CG2', 'GLY': 'CA'}
AA_TO_DISTAL.update({THREE_TO_ONE[aa]: v for aa, v in AA_TO_DISTAL.items()})

AA_TO_FUNCTIONAL = {'ARG': 'NH2', 'ASN': 'ND2', 'ASP': 'OD2', 'CYS': 'SG',
                    'GLN': 'NE2', 'GLU': 'OE2', 'HIS': 'NE2', 'ILE': 'CD1',
                    'LEU': 'CD2', 'LYS': 'NZ', 'MET': 'CE', 'PHE': 'CZ',
                    'PRO': 'CG', 'SER': 'OG', 'THR': 'OG1', 'TRP': 'CH2',
                    'TYR': 'CZ', 'VAL': 'CG2', 'GLY': 'CA'}
AA_TO_FUNCTIONAL.update({THREE_TO_ONE[aa]: v for aa, v in AA_TO_FUNCTIONAL.items()})

AA_TO_SC_ATOMS = {'MET': ['CB', 'CE', 'CG', 'SD'],
                  'ILE': ['CB', 'CD1', 'CG1', 'CG2'],
                  'LEU': ['CB', 'CD1', 'CD2', 'CG'],
                  'VAL': ['CB', 'CG1', 'CG2'],
                  'THR': ['CB', 'CG2', 'OG1'],
                  'ALA': ['CB'],
                  'ARG': ['CB', 'CD', 'CG', 'CZ', 'NE', 'NH1', 'NH2'],
                  'SER': ['CB', 'OG'],
                  'LYS': ['CB', 'CD', 'CE', 'CG', 'NZ'],
                  'HIS': ['CB', 'CD2', 'CE1', 'CG', 'ND1', 'NE2'],
                  'GLU': ['CB', 'CD', 'CG', 'OE1', 'OE2'],
                  'ASP': ['CB', 'CG', 'OD1', 'OD2'],
                  'PRO': ['CB', 'CD', 'CG'],
                  'GLN': ['CB', 'CD', 'CG', 'NE2', 'OE1'],
                  'TYR': ['CB', 'CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ', 'OH'],
                  'TRP': ['CB', 'CD1', 'CD2', 'CE2', 'CE3', 'CG', 'CH2', 'CZ2', 'CZ3', 'NE1'],
                  'CYS': ['CB', 'SG'],
                  'ASN': ['CB', 'CG', 'ND2', 'OD1'],
                  'PHE': ['CB', 'CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ'],
                  'GLY': []
                  }
AA_TO_SC_ATOMS.update({THREE_TO_ONE[aa]: v for aa, v in AA_TO_SC_ATOMS.items()})

to_posns = lambda chi: {THREE_TO_ONE[res]: [ALL_ATOM_POSNS[atom] for atom in chi[res]] for res in chi}


def to_chi_posns(chi):
    return {THREE_TO_ONE[res]: [ALL_ATOM_POSNS[atom] for atom in chi[res]] for res in chi}


CHI1 = {'PRO': ['N', 'CA', 'CB', 'CG'], 'THR': ['N', 'CA', 'CB', 'OG1'],
        'VAL': ['N', 'CA', 'CB', 'CG1'], 'LYS': ['N', 'CA', 'CB', 'CG'],
        'LEU': ['N', 'CA', 'CB', 'CG'], 'CYS': ['N', 'CA', 'CB', 'SG'],
        'HIS': ['N', 'CA', 'CB', 'CG'], 'MET': ['N', 'CA', 'CB', 'CG'],
        'ARG': ['N', 'CA', 'CB', 'CG'], 'SER': ['N', 'CA', 'CB', 'OG'],
        'TYR': ['N', 'CA', 'CB', 'CG'], 'PHE': ['N', 'CA', 'CB', 'CG'],
        'TRP': ['N', 'CA', 'CB', 'CG'], 'GLN': ['N', 'CA', 'CB', 'CG'],
        'ASN': ['N', 'CA', 'CB', 'CG'], 'ASP': ['N', 'CA', 'CB', 'CG'],
        'ILE': ['N', 'CA', 'CB', 'CG1'], 'GLU': ['N', 'CA', 'CB', 'CG']}
chi1_atom_posns = to_chi_posns(CHI1)

CHI2 = {'ARG': ['CA', 'CB', 'CG', 'CD'], 'ASN': ['CA', 'CB', 'CG', 'OD1'],
        'ASP': ['CA', 'CB', 'CG', 'OD1'], 'GLN': ['CA', 'CB', 'CG', 'CD'],
        'GLU': ['CA', 'CB', 'CG', 'CD'], 'HIS': ['CA', 'CB', 'CG', 'ND1'],
        'ILE': ['CA', 'CB', 'CG1', 'CD'], 'LEU': ['CA', 'CB', 'CG', 'CD1'],
        'LYS': ['CA', 'CB', 'CG', 'CD'], 'MET': ['CA', 'CB', 'CG', 'SD'],
        'PHE': ['CA', 'CB', 'CG', 'CD1'], 'PRO': ['CA', 'CB', 'CG', 'CD'],
        'TRP': ['CA', 'CB', 'CG', 'CD1'], 'TYR': ['CA', 'CB', 'CG', 'CD1']}
chi2_atom_posns = to_chi_posns(CHI2)

CHI3 = {'ARG': ['CB', 'CG', 'CD', 'NE'], 'GLN': ['CB', 'CG', 'CD', 'OE1'],
        'GLU': ['CB', 'CG', 'CD', 'OE1'], 'LYS': ['CB', 'CG', 'CD', 'CE'],
        'MET': ['CB', 'CG', 'SD', 'CE']}
chi3_atom_posns = to_chi_posns(CHI3)

CHI4 = {'ARG': ['CG', 'CD', 'NE', 'CZ'], 'LYS': ['CG', 'CD', 'CE', 'NZ']}
chi4_atom_posns = to_chi_posns(CHI4)

CHI5 = {'ARG': ['CD', 'NE', 'CZ', 'NH1']}
chi5_atom_posns = to_chi_posns(CHI5)

RES_TY_TO_CHI_MASK = torch.zeros(len(AA_ALPHABET), 4)
for aa in AA_ALPHABET:
    for i, chi in enumerate([CHI1, CHI2, CHI3, CHI4]):
        if aa not in ONE_TO_THREE:
            continue
        has_chi = 1 if ONE_TO_THREE[aa] in chi else 0
        RES_TY_TO_CHI_MASK[AA_INDEX_MAP[aa], i] = has_chi

chi_pi_periodic = {
    'ALA': [0.0, 0.0, 0.0, 0.0],
    'ARG': [0.0, 0.0, 0.0, 0.0],
    'ASN': [0.0, 0.0, 0.0, 0.0],
    'ASP': [0.0, 1.0, 0.0, 0.0],
    'CYS': [0.0, 0.0, 0.0, 0.0],
    'GLN': [0.0, 0.0, 0.0, 0.0],
    'GLU': [0.0, 0.0, 1.0, 0.0],
    'GLY': [0.0, 0.0, 0.0, 0.0],
    'HIS': [0.0, 0.0, 0.0, 0.0],
    'ILE': [0.0, 0.0, 0.0, 0.0],
    'LEU': [0.0, 0.0, 0.0, 0.0],
    'LYS': [0.0, 0.0, 0.0, 0.0],
    'MET': [0.0, 0.0, 0.0, 0.0],
    'PHE': [0.0, 1.0, 0.0, 0.0],
    'PRO': [0.0, 0.0, 0.0, 0.0],
    'SER': [0.0, 0.0, 0.0, 0.0],
    'THR': [0.0, 0.0, 0.0, 0.0],
    'TRP': [0.0, 0.0, 0.0, 0.0],
    'TYR': [0.0, 1.0, 0.0, 0.0],
    'VAL': [0.0, 0.0, 0.0, 0.0],
    'UNK': [0.0, 0.0, 0.0, 0.0],
}
chi_pi_periodic.update({THREE_TO_ONE[r]: arr for r, arr in chi_pi_periodic.items()})

SYMM_SC_RES_TYPES = ["ARG", "HIS", "ASP", "PHE", "GLN", "GLU", "LEU", "ASN",
                     "TYR", "VAL"]
SYMM_SC_RES_TYPES = [THREE_TO_ONE[x] for x in SYMM_SC_RES_TYPES]
SYMM_SC_RES_TYPE_SET = set(SYMM_SC_RES_TYPES)
SYMM_SC_RES_ATOMS = [[["NH1", "NH2"], ["NH1", "NH1"]],  # ARG *
                     [["ND1", "CD2"], ["CE1", "NE2"]],  # HIS - check
                     [["OD1", "OD2"], ["OD1", "OD1"]],  # ASP *
                     [["CD1", "CD2"], ["CE1", "CE2"]],  # PHE *
                     [["OE1", "NE2"], ["OE1", "OE1"]],  # GLN - check
                     [["OE1", "OE2"], ["OE1", "OE1"]],  # GLU *
                     [["CD1", "CD2"], ["CD1", "CD1"]],  # LEU - check
                     [["OD1", "ND2"], ["OD1", "OD1"]],  # ASN - check
                     [["CD1", "CD2"], ["CE1", "CE2"]],  # TYR *
                     [["CG1", "CG2"], ["CG1", "CG1"]],  # VAL - check
                     ]


class DSSPKeys:
    SS = 2
    AA = 1
    REL_ASA = 3
    PHI = 4
    PSI = 5
    SS_key_map = {'S': 1, 'H': 2, 'T': 3, 'I': 4, 'E': 5, 'G': 6, 'L': 7, 'B': 8, '-': 0}


# FROM https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2810841/
BOND_LENS = {("N", "C"): 1.33, ("CA", "CA"): 3.8, ("CA", "C"): 1.51, ("N", "CA"): 1.46}
BOND_LENS.update({(k[1], k[0]): v for k, v in BOND_LENS.items()})
BOND_LEN_SIGMA = 3
BOND_LEN_TOL = {("N", "C"): 0.01, ("CA", "CA"): 0.016, ("CA", "C"): 0.01, ("N", "CA"): 0.01}
BOND_LEN_TOL = {k: v * BOND_LEN_SIGMA for k, v in BOND_LEN_TOL.items()}
BOND_LEN_TOL.update({(k[1], k[0]): v for k, v in BOND_LEN_TOL.items()})
BOND_LEN_OFFSET = {("N", "C"): 1, ("CA", "CA"): 1, ("CA", "C"): 0, ("N", "CA"): 0}
BOND_LEN_OFFSET.update({(k[1], k[0]): v for k, v in BOND_LEN_OFFSET.items()})

BOND_ANGLE_SIGMA = 4
BOND_ANGLES = {("N", "CA", "C"): 111, ("CA", "C", "N"): 117.2, ("C", "N", "CA"): 121.7}
BOND_ANGLE_TOL = {("N", "CA", "C"): 2.8, ("CA", "C", "N"): 2.0, ("C", "N", "CA"): 1.8}
BOND_ANGLE_TOL = {k: v * BOND_ANGLE_SIGMA for k, v in BOND_ANGLE_TOL.items()}
BOND_ANGLE_OFFSET = {("N", "CA", "C"): (0, 0, 0), ("CA", "C", "N"): (0, 0, 1), ("C", "N", "CA"): (0, 1, 1)}

VDW_RADIUS = dict(C=1.7, O=1.52, N=1.55, S=1.8)
ALL_ATOM_VDW_RADII = {}
for atom in ALL_ATOMS:
    for ty in VDW_RADIUS:
        if ty in atom:
            ALL_ATOM_VDW_RADII[atom] = VDW_RADIUS[ty]
