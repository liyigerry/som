import networkx as nx
from rdkit import Chem
import matplotlib.pyplot as plt
import numpy as np

atom_type = ['C.3', 'C.2', 'C.1', 'C.ar', 'C.cat', 'N.3', 'N.2', 'N.1', 'N.ar', 'N.am',
            'N.pl3', 'N.4', 'O.3', 'O.2', 'O.co2', 'S.3', 'S.2', 'S.O', 'S.O2', 'H',
            'F', 'Cl', 'Br', 'I', 'LP', 'P.3', 'Na', 'K', 'Ca', 'Li', 'Al', 'Du', 'Si', 'Any']


path1 = r"C:\Users\Jane\Desktop\1_2_4_trichlorobenzene.sdf"
path2 = r"C:\Users\Jane\Desktop\2-acetylaminofluorene.sdf"

mol = Chem.SDMolSupplier(path2)[0]

ats = mol.GetProp('atomType').split()
print(ats)

# build graph

g = nx.Graph()
# add node
for atom in mol.GetAtoms():
    idx = atom.GetIdx()
    atomType = ats[idx]
    g.add_node(idx, AT=atomType)

# add edges
bonds_info = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
g.add_edges_from(bonds_info)
print(g.nodes[0]['AT'])
# show fig
plt.figure(figsize=(10, 8))
nx.draw_networkx(g, node_size=1000, width=3)
plt.show()


def get_neighbors(g, node, depth=1):
    output = {}
    output[0] = [node]
    layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1, depth+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x, []))
        nodes = output[i]
    return output


def nodes2vector(g, neighbor, depth=6):
    feature = []
    for i in range(depth):
        temp = [0 for i in range(len(atom_type))]
        if neighbor[i]:
            for j in neighbor[i]:
                if g.nodes[j]['AT'] in atom_type:
                    temp[atom_type.index(g.nodes[j]['AT'])] += 1
                else:
                    temp[-1] += 1
            feature.extend(temp)
        else:
            feature.extend(temp)
    return feature


for node in g.nodes():
    feature_vector = nodes2vector(g, get_neighbors(g, node, 6), depth=6)
    print(feature_vector)





