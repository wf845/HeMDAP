import numpy as np
import scipy.sparse as sp

####################################################
# This tool is to generate meta-path based adjacency
# matrix given original links.
####################################################

GM = np.genfromtxt("./data/gm-id.txt")
GD = np.genfromtxt("./data/gd-id.txt")

D = 435
M = 757
G = 11216
gm_ = sp.coo_matrix((np.ones(GM.shape[0]), (GM[:, 0], GM[:, 1])), shape=(G, M)).toarray()
gd_ = sp.coo_matrix((np.ones(GD.shape[0]), (GD[:, 0], GD[:, 1])), shape=(G, D)).toarray()

dgm = np.matmul(gd_.T, gm_) > 0
dgmgd = np.matmul(dgm, dgm.T) > 0
dgmgd = sp.coo_matrix(dgmgd)
sp.save_npz("./data/dgmgd.npz", dgmgd)