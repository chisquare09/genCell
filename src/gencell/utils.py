import numpy as np
from sklearn.metrics import pairwise_distances, rbf_kernel
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def compute_pdist(X, Y=None, metric='euclidean'):
    if Y is None:
        Y = X
    return pairwise_distances(X, Y, metric=metric, n_jobs=-1)


def energy_distance(X, Y):
    """Energy distance between two samples X and Y"""
    XX = compute_pdist(X)
    YY = compute_pdist(Y)
    XY = compute_pdist(X, Y)
    return np.sqrt(2*np.mean(XY) - np.mean(XX) - np.mean(YY))


def classwise_energy_distance(real_data, gen_data, labels, gen_labels):
    distances = {}
    for cls in np.unique(labels):
        real_cls = real_data[labels == cls]
        gen_cls = gen_data[gen_labels == cls]
        distances[cls] = energy_distance(real_cls, gen_cls)
    return distances


def spearman_per_class(real, real_labels, gen, gen_labels):
    cls_rhos = []
    for cls in np.unique(real_labels):
        R = real[ real_labels == cls ]
        G = gen [ gen_labels  == cls ]
        if R.shape[0] == 0 or G.shape[0] == 0:
            continue
        mu_r = np.mean(R, axis=0)
        mu_g = np.mean(G, axis=0)
        rho, _ = spearmanr(mu_r, mu_g)
        cls_rhos.append(rho)
    return np.nanmean(cls_rhos)


def mmd_per_class(real, real_labels, gen, gen_labels, gamma=None):
    def mmd(X, Y):
        γ = 1.0/X.shape[1] if gamma is None else gamma
        return (rbf_kernel(X,X,γ).mean() +
                rbf_kernel(Y,Y,γ).mean() -
                2*rbf_kernel(X,Y,γ).mean())
    scores = []
    for cls in np.unique(real_labels):
        r = real[ real_labels == cls ]
        g = gen  [ gen_labels  == cls ]
        scores.append(mmd(r,g))
    return np.mean(scores)


def rf_auc_per_class(real, real_labels, gen, gen_labels):
    aucs = []
    for cls in np.unique(real_labels):
        r = real[ real_labels == cls ]
        g = gen  [ gen_labels  == cls ]
        X = np.vstack([r, g])
        y = np.hstack([np.ones(len(r)), np.zeros(len(g))])
        rf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=0)
        rf.fit(X, y)
        p = rf.predict_proba(X)[:,1]
        aucs.append(roc_auc_score(y, p))
    return np.mean(aucs)


def knn_eval_per_class(real, real_labels, gen, gen_labels, k=5):
    accs, aucs = [], []
    for cls in np.unique(real_labels):
        r = real[ real_labels == cls ]
        g = gen  [ gen_labels  == cls ]
        X = np.vstack([r, g])
        y = np.hstack([np.ones(len(r)), np.zeros(len(g))])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(Xtr, ytr)
        ypred = knn.predict(Xte)
        yprob = knn.predict_proba(Xte)[:,1]
        accs.append(accuracy_score(yte, ypred))
        aucs.append(roc_auc_score(yte, yprob))
    return np.mean(accs), np.mean(aucs)


def evaluate_all(real_data, gen_data, real_labels, gen_labels, adata_real=None, adata_gen=None):
    
    print(f"Spearman SCC: {spearman_per_class(real_data, real_labels, gen_data, gen_labels):.4f}")
    print(f"MMD:          {mmd_per_class(real_data, real_labels, gen_data, gen_labels):.4f}")
    print(f"RF AUC:       {rf_auc_per_class(real_data, real_labels, gen_data, gen_labels):.4f}")
    acc, auc = knn_eval_per_class(real_data, real_labels, gen_data, gen_labels)
    print(f"KNN Acc:      {acc:.4f}, KNN AUC: {auc:.4f}")