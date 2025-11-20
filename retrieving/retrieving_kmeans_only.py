import os
import json
import pickle
import numpy as np
from sklearn.cluster import KMeans
from time import perf_counter


def load_pickle_embeddings(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    ids = data['ids']
    embs = np.array(data['embeddings'], dtype=np.float32)
    return ids, embs


def l2_normalize(mat: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


def build_kmeans_clusters(db_embs: np.ndarray, n_clusters: int = 10, random_state: int = 42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(db_embs)

    clusters = {}
    for cid in range(n_clusters):
        idxs = np.where(labels == cid)[0]
        clusters[cid] = idxs
    return kmeans, clusters


def retrieve_topk_cluster_only(
    val_embs: np.ndarray,
    kmeans: KMeans,
    clusters: dict,
    db_embs_norm: np.ndarray,
    topk: int = 1
):
    retrieved_indices = []
    times_sec = []

    for emb in val_embs:
        t0 = perf_counter()

        cid = int(kmeans.predict(emb.reshape(1, -1))[0])

        cluster_idxs = clusters.get(cid, np.array([], dtype=int))
        if cluster_idxs.size == 0:
            retrieved_indices.append([])
            times_sec.append(perf_counter() - t0)
            continue

        emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
        cluster_vecs = db_embs_norm[cluster_idxs]
        sims = cluster_vecs @ emb_norm.astype(np.float32)

        k = min(topk, cluster_idxs.size)
        if k <= 0:
            retrieved_indices.append([])
            times_sec.append(perf_counter() - t0)
            continue

        part = np.argpartition(sims, -k)[-k:]
        top_local = part[np.argsort(sims[part])[::-1]]
        top_global = cluster_idxs[top_local].tolist()

        dt = perf_counter() - t0
        retrieved_indices.append(top_global)
        times_sec.append(dt)

    return retrieved_indices, times_sec


def main(
    val_pkl='validation_embeddings_refined0824.pkl',
    db_pkl='database_embeddings_refined0824.pkl',
    train_json='3.5_refined_train0824.json',
    output_json='retrieval_results_top5_cluster_only_0825.json',
    topk=5,
    n_clusters=10
):
    val_ids, val_embs = load_pickle_embeddings(val_pkl)
    db_ids,  db_embs  = load_pickle_embeddings(db_pkl)

    with open(train_json, 'r', encoding='utf-8') as f:
        train_entries = json.load(f)

    kmeans, clusters = build_kmeans_clusters(db_embs, n_clusters=n_clusters, random_state=42)

    db_embs_norm = l2_normalize(db_embs.astype(np.float32))

    retrieved_indices, times_sec = retrieve_topk_cluster_only(
        val_embs=val_embs,
        kmeans=kmeans,
        clusters=clusters,
        db_embs_norm=db_embs_norm,
        topk=topk
    )

    total = len(times_sec)
    for i, (vid, t) in enumerate(zip(val_ids, times_sec), start=1):
        print(f"[{i}/{total}] retrieval for '{vid}': {t * 1000:.3f} ms")

    avg_ms = (sum(times_sec) / max(1, total)) * 1000.0
    print(f"Average retrieval time over {total} queries: {avg_ms:.3f} ms")

    results = []
    for vid, nbrs in zip(val_ids, retrieved_indices):
        scenes = [train_entries[idx] for idx in nbrs] if len(nbrs) > 0 else []
        results.append({
            'validation_id': vid,
            'retrieved_db_indices': nbrs,
            'retrieved_scenes': scenes
        })

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved full-scene retrieval results for {len(results)} validation items to '{output_json}'")


if __name__ == '__main__':
    main()