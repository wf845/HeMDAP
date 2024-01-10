import numpy as np
import scipy.sparse as sp
import torch as th
from sklearn.preprocessing import OneHotEncoder


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def load_d():
    # The order of node types: 0 p 1 a 2 s
    path = "./data/hmdd/"

    nei_d_g = np.load(path + "nei_d_g.npy", allow_pickle=True)
    nei_d_m = np.load(path + "nei_d_m.npy", allow_pickle=True)
    feat_d = sp.load_npz(path + "d_fea.npz")
    feat_m = sp.load_npz(path + "m_fea.npz")
    feat_g = sp.load_npz(path + "g_fea.npz")
    dmd = sp.load_npz(path + "dmd.npz")
    dgd = sp.load_npz(path + "dgd.npz")
    dgmgd = sp.load_npz(path + "dgmgd.npz")
    d_pos = sp.load_npz(path + "d_pos5.npz")

    nei_d_g = [th.LongTensor(i) for i in nei_d_g]
    nei_d_m = [th.LongTensor(i) for i in nei_d_m]
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_g = th.FloatTensor(preprocess_features(feat_g))
    dmd = sparse_mx_to_torch_sparse_tensor(normalize_adj(dmd))
    dgd = sparse_mx_to_torch_sparse_tensor(normalize_adj(dgd))
    dgmgd = sparse_mx_to_torch_sparse_tensor(normalize_adj(dgmgd))
    pos = sparse_mx_to_torch_sparse_tensor(d_pos)

    return [nei_d_m, nei_d_g], [feat_d, feat_m, feat_g], [dmd, dgd, dgmgd], pos


def load_m():
    # The order of node types: 0 p 1 a 2 s
    path = "./data/hmdd/"

    nei_m_d = np.load(path + "nei_m_d.npy", allow_pickle=True)
    nei_m_g = np.load(path + "nei_m_g.npy", allow_pickle=True)
    feat_d = sp.load_npz(path + "d_fea.npz")
    feat_m = sp.load_npz(path + "m_fea.npz")
    feat_g = sp.load_npz(path + "g_fea.npz")
    mm = sp.load_npz(path + "mm.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mgm = sp.load_npz(path + "mgm.npz")
    mgdgm = sp.load_npz(path + "mgdgm.npz")
    m_pos = sp.load_npz(path + "m_pos5.npz")

    nei_m_d = [th.LongTensor(i) for i in nei_m_d]
    nei_m_g = [th.LongTensor(i) for i in nei_m_g]
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_g = th.FloatTensor(preprocess_features(feat_g))
    mm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mm))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    mgm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mgm))
    mgdgm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mgdgm))
    pos = sparse_mx_to_torch_sparse_tensor(m_pos)

    return [nei_m_d, nei_m_g], [feat_m, feat_d, feat_g], [mm, mdm, mgm, mgdgm], pos
