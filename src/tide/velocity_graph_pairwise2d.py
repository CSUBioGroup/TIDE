# velocity_graph_pairwise2d.py
# Drop-in scVelo-like velocity graph with pairwise 2D gene-wise TopK aggregation
# + branch-robust gene lists via edge-weighted aggregation over outgoing edges.
# Adapted from the project scVelo fork; see NOTICE.

import os
import numpy as np
from scipy.sparse import coo_matrix, issparse

from scvelo import logging as logg
from scvelo import settings
from scvelo.core import get_n_jobs, l2_norm, parallelize
from scvelo.preprocessing.moments import get_moments
from scvelo.preprocessing.neighbors import (
    get_n_neighs,
    get_neighs,
    neighbors,
    pca,
    verify_neighbors,
)
from scvelo.tools.utils import cosine_correlation, get_indices, get_iterative_indices
from scvelo.tools.velocity import velocity


def vals_to_csr(vals, rows, cols, shape, split_negative=False):
    """Convert edge lists to CSR; optionally split negative edges."""
    graph = coo_matrix((vals, (rows, cols)), shape=shape)

    if split_negative:
        graph_neg = graph.copy()
        graph.data = np.clip(graph.data, 0, 1)
        graph_neg.data = np.clip(graph_neg.data, -1, 0)

        graph.eliminate_zeros()
        graph_neg.eliminate_zeros()
        return graph.tocsr(), graph_neg.tocsr()
    else:
        return graph.tocsr()


def _flatten(iterable):
    return [i for it in iterable for i in it]


class VelocityGraph:
    """Velocity graph computation (scVelo style) with extra graph modes."""

    def __init__(
        self,
        adata,
        vkey="velocity",
        xkey="Ms",
        tkey=None,
        basis=None,
        n_neighbors=None,
        sqrt_transform=None,
        n_recurse_neighbors=None,
        random_neighbors_at_max=None,
        gene_subset=None,
        approx=None,
        report=False,
        compute_uncertainties=None,
        mode_neighbors="distances",
        # -------- graph mode --------
        graph_mode="pairwise2d_geneTopK",  # "original" or "pairwise2d_geneTopK"
        # -------- 2D inputs --------
        ukey="Mu",
        skey=None,
        vkey_u=None,
        eps=1e-8,
        # -------- weights --------
        weight_mode="speed",  # "speed" or "uniform"
        # -------- pairwise TopK params --------
        pairwise_topk_genes=50,
        pairwise_candidate_genes=400,   # IMPORTANT for speed
        pairwise_use_signed_cos=True,
        # -------- NEW: store gene stats (branch-aware aggregation) --------
        store_gene_stats=False,
        store_k=100,
        store_edge_topm=30,             # only use top-M outgoing positive edges for gene aggregation
        store_edge_weight="sim_pos",    # "sim_pos" or "uniform"
    ):
        self.compute_uncertainties = compute_uncertainties
        self.uncertainties = None
        self.self_prob = None

        self.graph_mode = str(graph_mode)
        self.ukey = str(ukey)
        self.skey = xkey if skey is None else str(skey)
        self.vkey_u = vkey_u
        self.eps = float(eps)
        self.weight_mode = str(weight_mode)

        self.pairwise_topk_genes = int(pairwise_topk_genes) if pairwise_topk_genes is not None else None
        self.pairwise_candidate_genes = int(pairwise_candidate_genes) if pairwise_candidate_genes is not None else None
        self.pairwise_use_signed_cos = bool(pairwise_use_signed_cos)

        self.store_gene_stats = bool(store_gene_stats)
        self.store_k = int(store_k)
        self.store_edge_topm = int(store_edge_topm) if store_edge_topm is not None else None
        self.store_edge_weight = str(store_edge_weight)

        self.report = report
        self.adata = adata

        # filled when store_gene_stats=True
        self.gene_names_pairwise = None
        self.gene_stats = None

        subset = np.ones(adata.n_vars, bool)
        if gene_subset is not None:
            var_names_subset = adata.var_names.isin(gene_subset)
            subset &= var_names_subset if len(var_names_subset) > 0 else gene_subset
        elif f"{vkey}_genes" in adata.var.keys():
            subset &= np.array(adata.var[f"{vkey}_genes"].values, dtype=bool)

        xkey = xkey if xkey in adata.layers.keys() else "spliced"

        # ---------- original scVelo path arrays (X, V) ----------
        X = np.array(
            adata.layers[xkey].toarray()[:, subset]
            if issparse(adata.layers[xkey])
            else adata.layers[xkey][:, subset]
        )
        V = np.array(
            adata.layers[vkey].toarray()[:, subset]
            if issparse(adata.layers[vkey])
            else adata.layers[vkey][:, subset]
        )

        nans = np.isnan(np.sum(V, axis=0))
        if np.any(nans):
            X = X[:, ~nans]
            V = V[:, ~nans]

        if approx is True and X.shape[1] > 100:
            X_pca, PCs, _, _ = pca(X, n_comps=30, svd_solver="arpack", return_info=True)
            self.X = np.array(X_pca, dtype=np.float32)
            self.V = (V - V.mean(0)).dot(PCs.T)
            self.V[self.V.sum(1) == 0] = 0
        else:
            self.X = np.array(X, dtype=np.float32)
            self.V = np.array(V, dtype=np.float32)

        self.V_raw = np.array(self.V)

        self.sqrt_transform = sqrt_transform
        uns_key = f"{vkey}_params"
        if self.sqrt_transform is None:
            if uns_key in adata.uns.keys() and "mode" in adata.uns[uns_key]:
                self.sqrt_transform = adata.uns[uns_key]["mode"] == "stochastic"
        if self.sqrt_transform:
            self.V = np.sqrt(np.abs(self.V)) * np.sign(self.V)
        self.V -= np.nanmean(self.V, axis=1)[:, None]

        # ---------- 2D arrays for pairwise2d --------
        self.U = None
        self.S = None
        self.Vu = None
        self.Vs = None

        if self.graph_mode == "pairwise2d_geneTopK":
            if self.skey not in adata.layers.keys():
                raise ValueError(f"[pairwise2d_geneTopK] skey='{self.skey}' not found in adata.layers.")
            if self.ukey not in adata.layers.keys():
                if "Mu" in adata.layers.keys():
                    self.ukey = "Mu"
                elif "unspliced" in adata.layers.keys():
                    self.ukey = "unspliced"
                else:
                    raise ValueError("[pairwise2d_geneTopK] Need unspliced layer. Tried ukey='Mu' and 'unspliced'.")

            if self.vkey_u is None:
                cand1 = f"{vkey}_u"
                if cand1 in adata.layers.keys():
                    self.vkey_u = cand1
                elif "velocity_u" in adata.layers.keys():
                    self.vkey_u = "velocity_u"
                else:
                    raise ValueError(
                        "[pairwise2d_geneTopK] Need unspliced velocity layer.\n"
                        f"Tried '{cand1}' and 'velocity_u'."
                    )
            if self.vkey_u not in adata.layers.keys():
                raise ValueError(f"[pairwise2d_geneTopK] vkey_u='{self.vkey_u}' not found in adata.layers.")

            if self.compute_uncertainties:
                raise NotImplementedError("[pairwise2d_geneTopK] compute_uncertainties is not supported.")

            def _dense(layer_key):
                mat = adata.layers[layer_key]
                return mat.toarray() if issparse(mat) else np.asarray(mat)

            U0 = _dense(self.ukey)[:, subset].astype(np.float32)
            S0 = _dense(self.skey)[:, subset].astype(np.float32)
            Vu0 = _dense(self.vkey_u)[:, subset].astype(np.float32)
            Vs0 = _dense(vkey)[:, subset].astype(np.float32)

            bad = (
                np.isnan(np.sum(Vu0, axis=0))
                | np.isnan(np.sum(Vs0, axis=0))
                | np.isnan(np.sum(U0, axis=0))
                | np.isnan(np.sum(S0, axis=0))
            )
            if np.any(bad):
                U0, S0, Vu0, Vs0 = U0[:, ~bad], S0[:, ~bad], Vu0[:, ~bad], Vs0[:, ~bad]

            # gene names for stats (aligned to 2D arrays)
            if self.store_gene_stats:
                var_names = np.asarray(adata.var_names)[subset]
                if np.any(bad):
                    var_names = var_names[~bad]
                self.gene_names_pairwise = var_names.astype(str)

            # hard sanitize
            U0  = np.nan_to_num(U0,  nan=0.0, posinf=0.0, neginf=0.0)
            S0  = np.nan_to_num(S0,  nan=0.0, posinf=0.0, neginf=0.0)
            Vu0 = np.nan_to_num(Vu0, nan=0.0, posinf=0.0, neginf=0.0)
            Vs0 = np.nan_to_num(Vs0, nan=0.0, posinf=0.0, neginf=0.0)

            if self.sqrt_transform:
                Vu0 = np.sqrt(np.abs(Vu0)) * np.sign(Vu0)
                Vs0 = np.sqrt(np.abs(Vs0)) * np.sign(Vs0)

            Vu0 = Vu0 - np.nanmean(Vu0, axis=1, keepdims=True)
            Vs0 = Vs0 - np.nanmean(Vs0, axis=1, keepdims=True)

            self.U = U0
            self.S = S0
            self.Vu = Vu0
            self.Vs = Vs0

        # ---------- neighbor indices ----------
        self.n_recurse_neighbors = n_recurse_neighbors
        if self.n_recurse_neighbors is None:
            if n_neighbors is not None or mode_neighbors == "connectivities":
                self.n_recurse_neighbors = 1
            else:
                self.n_recurse_neighbors = 2

        if "neighbors" not in adata.uns.keys():
            neighbors(adata)
        if np.min((get_neighs(adata, "distances") > 0).sum(1).A1) == 0:
            raise ValueError("Your neighbor graph seems corrupted. Consider recomputing via pp.neighbors.")

        if n_neighbors is None or n_neighbors <= get_n_neighs(adata):
            self.indices = get_indices(
                dist=get_neighs(adata, "distances"),
                n_neighbors=n_neighbors,
                mode_neighbors=mode_neighbors,
            )[0]
        else:
            if basis is None:
                basis_keys = ["X_pca", "X_tsne", "X_umap"]
                basis = [key for key in basis_keys if key in adata.obsm.keys()][-1]
            elif f"X_{basis}" in adata.obsm.keys():
                basis = f"X_{basis}"

            if isinstance(approx, str) and approx in adata.obsm.keys():
                from sklearn.neighbors import NearestNeighbors
                neighs = NearestNeighbors(n_neighbors=n_neighbors + 1)
                neighs.fit(adata.obsm[approx])
                self.indices = neighs.kneighbors_graph(mode="connectivity").indices.reshape((-1, n_neighbors + 1))
            else:
                from scvelo import Neighbors
                neighs = Neighbors(adata)
                neighs.compute_neighbors(n_neighbors=n_neighbors, use_rep=basis, n_pcs=10)
                self.indices = get_indices(dist=neighs.distances, mode_neighbors=mode_neighbors)[0]

        self.max_neighs = random_neighbors_at_max

        gkey, gkey_ = f"{vkey}_graph", f"{vkey}_graph_neg"
        self.graph = adata.uns[gkey] if gkey in adata.uns.keys() else []
        self.graph_neg = adata.uns[gkey_] if gkey_ in adata.uns.keys() else []

        # tkey (time groups) support
        if tkey in adata.obs.keys():
            self.t0 = adata.obs[tkey].astype("category").copy()
            init = min(self.t0) if isinstance(min(self.t0), int) else 0
            self.t0 = self.t0.cat.set_categories(np.arange(init, len(self.t0.cat.categories)), rename=True)
            self.t1 = self.t0.copy()
            self.t1 = self.t1.cat.set_categories(self.t0.cat.categories + 1, rename=True)
        else:
            self.t0 = None

    def _select_outgoing_edges_for_stats(self, sim):
        """
        Return keep_edges (indices into neighbor list) and edge weights.
        - keep only outgoing (positive) edges
        - then take top-M by edge weight (or by sim if uniform)
        """
        eps = self.eps
        if sim.size == 0:
            return None, None

        if self.store_edge_weight == "sim_pos":
            edge_w = np.clip(sim, 0.0, 1.0).astype(np.float32)
            pos_mask = edge_w > 0
        elif self.store_edge_weight == "uniform":
            edge_w = np.ones_like(sim, dtype=np.float32)
            pos_mask = sim > 0
        else:
            raise ValueError("[pairwise2d_geneTopK] store_edge_weight must be 'sim_pos' or 'uniform'.")

        if not np.any(pos_mask):
            return None, None

        cand_edges = np.where(pos_mask)[0]
        if cand_edges.size == 0:
            return None, None

        M = self.store_edge_topm
        if M is None or M <= 0 or cand_edges.size <= M:
            keep_edges = cand_edges
        else:
            # topM among outgoing edges
            ww = edge_w[cand_edges]
            sel = np.argpartition(ww, -M)[-M:]
            keep_edges = cand_edges[sel]
            keep_edges = keep_edges[np.argsort(edge_w[keep_edges])[::-1]]

        ew = edge_w[keep_edges].astype(np.float32)
        if not np.isfinite(ew).all():
            ew = np.nan_to_num(ew, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if ew.sum() <= eps:
            return None, None
        return keep_edges, ew

    def _gene_stats_weighted_edges(self, sim, cos_ijg, idx_cand):
        """
        B) cos_{i,g} = weighted_mean_{j in keep_edges} cos_{ijg}, weights = edge_w.
        - keep_edges: top-M outgoing positive edges
        - genes: candidate genes only (idx_cand)
        """
        keep_edges, ew = self._select_outgoing_edges_for_stats(sim)
        if keep_edges is None:
            return None

        # cos_ijg: (nnb, C)
        sub = cos_ijg[keep_edges]  # (m, C)
        # weighted mean across edges
        num = ew @ sub  # (C,)
        den = float(ew.sum()) + self.eps
        cos_g = (num / den).astype(np.float32)
        cos_g = np.nan_to_num(cos_g, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        C = cos_g.shape[0]
        if C == 0:
            return None

        Ksave = max(1, min(int(self.store_k), C))

        if Ksave < C:
            idx_max_c = np.argpartition(cos_g, -Ksave)[-Ksave:]
            idx_max_c = idx_max_c[np.argsort(cos_g[idx_max_c])[::-1]]

            idx_min_c = np.argpartition(cos_g, Ksave - 1)[:Ksave]
            idx_min_c = idx_min_c[np.argsort(cos_g[idx_min_c])]  # ascending

            abs_cos = np.abs(cos_g)
            idx_0_c = np.argpartition(abs_cos, Ksave - 1)[:Ksave]
            idx_0_c = idx_0_c[np.argsort(abs_cos[idx_0_c])]
        else:
            idx_max_c = np.argsort(cos_g)[::-1]
            idx_min_c = np.argsort(cos_g)
            idx_0_c = np.argsort(np.abs(cos_g))

        # map candidate-local indices -> global gene indices (in filtered 2D gene space)
        max_idx = idx_cand[idx_max_c].astype(np.int32)
        min_idx = idx_cand[idx_min_c].astype(np.int32)
        near0_idx = idx_cand[idx_0_c].astype(np.int32)

        gstat = {
            "max_idx": max_idx[:Ksave],
            "max_cos": cos_g[idx_max_c[:Ksave]].astype(np.float32),
            "min_idx": min_idx[:Ksave],
            "min_cos": cos_g[idx_min_c[:Ksave]].astype(np.float32),
            "near0_idx": near0_idx[:Ksave],
            "near0_cos": cos_g[idx_0_c[:Ksave]].astype(np.float32),
        }
        return gstat

    def _cosine_pairwise2d_geneTopK(self, obs_id, neighs_idx, return_gene_stats=False):
        """
        Pair-wise 2D gene cosine:
          cos_ijg = cos( unit([Δu_ijg, Δs_ijg]), unit([vu_ig, vs_ig]) )
        Edge score:
          π_ij = weighted_mean( topK_g cos_ijg )

        Gene stats (when return_gene_stats=True):
          B) cos_{i,g} = weighted_mean_{j in topM outgoing} cos_ijg,
             weights = edge_w (sim_pos or uniform), genes = candidate genes only.
        """
        eps = self.eps
        U, S, Vu, Vs = self.U, self.S, self.Vu, self.Vs
        nnb = len(neighs_idx)
        sim0 = np.zeros(nnb, dtype=np.float32)
        if U is None or S is None or Vu is None or Vs is None or nnb == 0:
            return (sim0, None) if return_gene_stats else sim0

        vu = Vu[obs_id].astype(np.float32)
        vs = Vs[obs_id].astype(np.float32)
        speed = np.sqrt(vu * vu + vs * vs).astype(np.float32)

        inv = np.divide(1.0, speed, out=np.zeros_like(speed), where=(speed > eps))
        ux = vu * inv
        uy = vs * inv

        if self.weight_mode == "speed":
            w = speed
        elif self.weight_mode == "uniform":
            w = np.ones_like(speed, dtype=np.float32)
        else:
            raise ValueError("[pairwise2d_geneTopK] weight_mode must be 'speed' or 'uniform'.")

        G = ux.shape[0]

        # candidate genes (speed-based) to reduce cost
        C = self.pairwise_candidate_genes
        if C is None:
            C = G
        C = max(1, min(int(C), G))
        if C < G:
            idx_cand = np.argpartition(w, -C)[-C:]
        else:
            idx_cand = np.arange(G, dtype=np.int32)

        # per-edge TopK within candidates (for EDGE sim only)
        K = self.pairwise_topk_genes
        if K is None:
            K = min(50, C)
        K = max(1, min(int(K), C))

        # gather candidates
        ux_c = ux[idx_cand]
        uy_c = uy[idx_cand]
        w_c = w[idx_cand].astype(np.float32)

        Ui = U[obs_id, idx_cand].astype(np.float32)
        Si = S[obs_id, idx_cand].astype(np.float32)

        Un = U[neighs_idx][:, idx_cand].astype(np.float32)  # (nnb,C)
        Sn = S[neighs_idx][:, idx_cand].astype(np.float32)

        dU = Un - Ui[None, :]
        dS = Sn - Si[None, :]

        if self.sqrt_transform:
            dU = np.sqrt(np.abs(dU)) * np.sign(dU)
            dS = np.sqrt(np.abs(dS)) * np.sign(dS)

        dn = np.sqrt(dU * dU + dS * dS).astype(np.float32)
        invd = np.divide(1.0, dn, out=np.zeros_like(dn), where=(dn > eps))
        dxu = dU * invd
        dxs = dS * invd

        cos_ijg = dxu * ux_c[None, :] + dxs * uy_c[None, :]
        cos_ijg = np.nan_to_num(cos_ijg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # NOTE:
        # If you clip to [0,1], then "min" will collapse toward 0 and can overlap with near0 by definition.
        if not self.pairwise_use_signed_cos:
            cos_ijg = np.clip(cos_ijg, 0.0, 1.0)

        # per-edge topK aggregation -> edge sim
        if K < C:
            idx_top = np.argpartition(cos_ijg, -K, axis=1)[:, -K:]  # (nnb,K)
            cos_sel = np.take_along_axis(cos_ijg, idx_top, axis=1).astype(np.float32)
            w_sel = w_c[idx_top].astype(np.float32)
        else:
            cos_sel = cos_ijg.astype(np.float32)
            w_sel = np.broadcast_to(w_c[None, :], cos_sel.shape).astype(np.float32)

        num = (w_sel * cos_sel).sum(axis=1).astype(np.float32)
        den = (w_sel).sum(axis=1).astype(np.float32) + eps
        sim = num / den
        sim[~np.isfinite(sim)] = 0.0

        if not return_gene_stats:
            return sim

        # ---- B) edge-weighted mean across outgoing edges for each gene (candidate genes only) ----
        gstat = self._gene_stats_weighted_edges(sim=sim, cos_ijg=cos_ijg, idx_cand=idx_cand)
        return sim, gstat

    def compute_cosines(self, n_jobs=None, backend="loky", show_progress_bar: bool = True):
        n_jobs = get_n_jobs(n_jobs=n_jobs)
        n_obs = self.X.shape[0]

        res = parallelize(
            self._compute_cosines,
            range(n_obs),
            n_jobs=n_jobs,
            unit="cells",
            backend=backend,
            as_array=False,
            show_progress_bar=show_progress_bar,
        )()

        uncertainties, vals, rows, cols, gene_stats = map(_flatten, zip(*res))

        vals = np.hstack(vals) if len(vals) > 0 else np.array([], dtype=np.float32)
        vals[np.isnan(vals)] = 0.0

        self.graph, self.graph_neg = vals_to_csr(vals, rows, cols, shape=(n_obs, n_obs), split_negative=True)

        # scvelo's self_transition heuristic
        confidence = self.graph.max(1).toarray().flatten()
        self.self_prob = np.clip(np.percentile(confidence, 98) - confidence, 0, 1)

        # store gene stats (per-cell)
        if self.graph_mode == "pairwise2d_geneTopK" and self.store_gene_stats:
            Ksave = int(self.store_k)
            max_idx = np.full((n_obs, Ksave), -1, dtype=np.int32)
            max_cos = np.full((n_obs, Ksave), np.nan, dtype=np.float32)
            min_idx = np.full((n_obs, Ksave), -1, dtype=np.int32)
            min_cos = np.full((n_obs, Ksave), np.nan, dtype=np.float32)
            near0_idx = np.full((n_obs, Ksave), -1, dtype=np.int32)
            near0_cos = np.full((n_obs, Ksave), np.nan, dtype=np.float32)

            for obs_id, gs in gene_stats:
                if gs is None:
                    continue

                k1 = min(Ksave, len(gs["max_idx"]))
                max_idx[obs_id, :k1] = gs["max_idx"][:k1]
                max_cos[obs_id, :k1] = gs["max_cos"][:k1]

                k2 = min(Ksave, len(gs["min_idx"]))
                min_idx[obs_id, :k2] = gs["min_idx"][:k2]
                min_cos[obs_id, :k2] = gs["min_cos"][:k2]

                k3 = min(Ksave, len(gs["near0_idx"]))
                near0_idx[obs_id, :k3] = gs["near0_idx"][:k3]
                near0_cos[obs_id, :k3] = gs["near0_cos"][:k3]

            self.gene_stats = {
                "genes": None if self.gene_names_pairwise is None else np.asarray(self.gene_names_pairwise, dtype=str),
                "max_idx": max_idx,
                "max_cos": max_cos,
                "min_idx": min_idx,
                "min_cos": min_cos,
                "near0_idx": near0_idx,
                "near0_cos": near0_cos,
                "store_k": int(Ksave),
                "graph_mode": str(self.graph_mode),
                "pairwise_topk_genes": int(self.pairwise_topk_genes),
                "pairwise_candidate_genes": int(self.pairwise_candidate_genes),
                "pairwise_use_signed_cos": bool(self.pairwise_use_signed_cos),
                "weight_mode": str(self.weight_mode),
                "store_edge_topm": int(self.store_edge_topm) if self.store_edge_topm is not None else None,
                "store_edge_weight": str(self.store_edge_weight),
                "ukey": str(self.ukey),
                "skey": str(self.skey),
                "vkey_u": None if self.vkey_u is None else str(self.vkey_u),
                "gene_cosine_def": "cos_i,g = weighted_mean_{j in topM outgoing} cos_ijg, weights=edge_w (candidate genes only)",
            }
        else:
            self.gene_stats = None

    def _compute_cosines(self, obs_idx, queue):
        gene_stats = []
        vals, rows, cols, uncertainties = [], [], [], []

        if self.compute_uncertainties:
            moments = get_moments(self.adata, np.sign(self.V_raw), second_order=True)

        for obs_id in obs_idx:
            if self.graph_mode == "pairwise2d_geneTopK":
                ok = False if (self.Vu is None or self.Vs is None) else (
                    self.Vu[obs_id].max() != 0 or self.Vu[obs_id].min() != 0 or
                    self.Vs[obs_id].max() != 0 or self.Vs[obs_id].min() != 0
                )
            else:
                ok = (self.V[obs_id].max() != 0 or self.V[obs_id].min() != 0)

            if ok:
                neighs_idx = get_iterative_indices(self.indices, obs_id, self.n_recurse_neighbors, self.max_neighs)

                if self.t0 is not None:
                    t0, t1 = self.t0[obs_id], self.t1[obs_id]
                    if t0 >= 0 and t1 > 0:
                        t1_idx = np.where(self.t0 == t1)[0]
                        if len(t1_idx) > len(neighs_idx):
                            t1_idx = np.random.choice(t1_idx, len(neighs_idx), replace=False)
                        if len(t1_idx) > 0:
                            neighs_idx = np.unique(np.concatenate([neighs_idx, t1_idx]))

                if self.graph_mode == "pairwise2d_geneTopK":
                    if self.store_gene_stats:
                        val, gstat = self._cosine_pairwise2d_geneTopK(
                            obs_id, neighs_idx, return_gene_stats=True
                        )
                        if gstat is not None:
                            gene_stats.append((obs_id, gstat))
                    else:
                        val = self._cosine_pairwise2d_geneTopK(obs_id, neighs_idx, return_gene_stats=False)
                else:
                    dX = self.X[neighs_idx] - self.X[obs_id, None]
                    if self.sqrt_transform:
                        dX = np.sqrt(np.abs(dX)) * np.sign(dX)
                    val = cosine_correlation(dX, self.V[obs_id])

                    if self.compute_uncertainties:
                        dX /= l2_norm(dX)[:, None]
                        uncertainties.extend(np.nansum(dX**2 * moments[obs_id][None, :], 1))

                vals.extend(val.tolist())
                rows.extend([obs_id] * len(neighs_idx))
                cols.extend(neighs_idx.tolist())

            if queue is not None:
                queue.put(1)

        if queue is not None:
            queue.put(None)

        return uncertainties, vals, rows, cols, gene_stats


def velocity_graph(
    data,
    vkey="velocity",
    xkey="Ms",
    tkey=None,
    basis=None,
    n_neighbors=None,
    n_recurse_neighbors=None,
    random_neighbors_at_max=None,
    sqrt_transform=None,
    variance_stabilization=None,
    gene_subset=None,
    compute_uncertainties=None,
    approx=None,
    mode_neighbors="distances",
    copy=False,
    n_jobs=None,
    backend="loky",
    show_progress_bar: bool = True,
    # -------- graph mode --------
    graph_mode="pairwise2d_geneTopK",  # default for this file
    # -------- 2D inputs --------
    ukey="Mu",
    skey=None,
    vkey_u=None,
    eps=1e-8,
    # -------- weights --------
    weight_mode="speed",
    # -------- pairwise params --------
    pairwise_topk_genes=50,
    pairwise_candidate_genes=400,
    pairwise_use_signed_cos=True,
    # -------- NEW: store gene stats --------
    store_gene_stats=False,
    store_k=100,
    store_edge_topm=30,
    store_edge_weight="sim_pos",
):
    """
    scVelo-like velocity_graph with additional mode:
      - graph_mode="pairwise2d_geneTopK"

    If store_gene_stats=True:
      adata.uns[f"{vkey}_gene_cosine_stats"] will be saved.
      It contains per-cell HIGH/LOW/NEAR0 gene indices and corresponding cos_i,g values
      defined as edge-weighted mean over outgoing edges.
    """
    adata = data.copy() if copy else data
    verify_neighbors(adata)

    if vkey not in adata.layers.keys():
        velocity(adata, vkey=vkey)

    if sqrt_transform is None:
        sqrt_transform = variance_stabilization

    vgraph = VelocityGraph(
        adata,
        vkey=vkey,
        xkey=xkey,
        tkey=tkey,
        basis=basis,
        n_neighbors=n_neighbors,
        approx=approx,
        n_recurse_neighbors=n_recurse_neighbors,
        random_neighbors_at_max=random_neighbors_at_max,
        sqrt_transform=sqrt_transform,
        gene_subset=gene_subset,
        compute_uncertainties=compute_uncertainties,
        report=True,
        mode_neighbors=mode_neighbors,
        graph_mode=graph_mode,
        ukey=ukey,
        skey=skey,
        vkey_u=vkey_u,
        eps=eps,
        weight_mode=weight_mode,
        pairwise_topk_genes=pairwise_topk_genes,
        pairwise_candidate_genes=pairwise_candidate_genes,
        pairwise_use_signed_cos=pairwise_use_signed_cos,
        store_gene_stats=store_gene_stats,
        store_k=store_k,
        store_edge_topm=store_edge_topm,
        store_edge_weight=store_edge_weight,
    )

    if isinstance(basis, str):
        logg.warn(
            f"The velocity graph is computed on {basis} embedding coordinates.\n"
            f"        Consider computing the graph in an unbiased manner \n"
            f"        on full expression space by not specifying basis.\n"
        )

    n_jobs = get_n_jobs(n_jobs=n_jobs)
    logg.info(f"computing velocity graph (using {n_jobs}/{os.cpu_count()} cores)", r=True)
    vgraph.compute_cosines(n_jobs=n_jobs, backend=backend, show_progress_bar=show_progress_bar)

    # save graph
    adata.uns[f"{vkey}_graph"] = vgraph.graph
    adata.uns[f"{vkey}_graph_neg"] = vgraph.graph_neg
    adata.obs[f"{vkey}_self_transition"] = vgraph.self_prob

    # save gene stats (per-cell lists)
    if graph_mode == "pairwise2d_geneTopK" and store_gene_stats:
        adata.uns[f"{vkey}_gene_cosine_stats"] = vgraph.gene_stats

    # params
    if f"{vkey}_params" in adata.uns.keys():
        if "embeddings" in adata.uns[f"{vkey}_params"]:
            del adata.uns[f"{vkey}_params"]["embeddings"]
    else:
        adata.uns[f"{vkey}_params"] = {}

    adata.uns[f"{vkey}_params"]["mode_neighbors"] = mode_neighbors
    adata.uns[f"{vkey}_params"]["n_recurse_neighbors"] = vgraph.n_recurse_neighbors
    adata.uns[f"{vkey}_params"]["graph_mode"] = graph_mode

    if graph_mode == "pairwise2d_geneTopK":
        adata.uns[f"{vkey}_params"]["pairwise_topk_genes"] = int(pairwise_topk_genes)
        adata.uns[f"{vkey}_params"]["pairwise_candidate_genes"] = int(pairwise_candidate_genes)
        adata.uns[f"{vkey}_params"]["pairwise_use_signed_cos"] = bool(pairwise_use_signed_cos)
        adata.uns[f"{vkey}_params"]["weight_mode"] = str(weight_mode)
        adata.uns[f"{vkey}_params"]["ukey"] = str(ukey)
        adata.uns[f"{vkey}_params"]["skey"] = str(xkey if skey is None else skey)
        adata.uns[f"{vkey}_params"]["vkey_u"] = str(vgraph.vkey_u)
        adata.uns[f"{vkey}_params"]["store_gene_stats"] = bool(store_gene_stats)
        adata.uns[f"{vkey}_params"]["store_k"] = int(store_k)
        adata.uns[f"{vkey}_params"]["store_edge_topm"] = int(store_edge_topm)
        adata.uns[f"{vkey}_params"]["store_edge_weight"] = str(store_edge_weight)

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint("added \n" f"    '{vkey}_graph', sparse matrix with cosine correlations (adata.uns)")

    return adata if copy else None
