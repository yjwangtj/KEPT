import os
import json
import pickle
import numpy as np
import hnswlib
from sklearn.cluster import KMeans
from time import perf_counter


def load_pickle_embeddings(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    ids = data['ids']
    embs = np.array(data['embeddings'], dtype=np.float32)
    return ids, embs


def build_hnsw_indices(db_embs, n_clusters=10, dim=128,
                       ef_construction=200, M=16, ef_search=50):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(db_embs)
    labels = kmeans.labels_
    indices = {}
    for cid in range(n_clusters):
        idxs = np.where(labels == cid)[0]
        if len(idxs) == 0:
            continue
        p = hnswlib.Index(space='cosine', dim=dim)
        p.init_index(max_elements=len(idxs), ef_construction=ef_construction, M=M)
        p.add_items(db_embs[idxs], idxs.astype(np.int32))
        p.set_ef(ef_search)
        indices[cid] = p
    return kmeans, indices


def retrieve_topk(val_embs, kmeans, hnsw_indices, topk=1):
    retrieved = []
    times_sec = []

    for emb in val_embs:
        cid = int(kmeans.predict(emb.reshape(1, -1))[0])
        index = hnsw_indices.get(cid, None)

        if index is None or index.get_current_count() == 0:
            retrieved.append([])
            times_sec.append(0.0)
            continue

        k_eff = min(topk, index.get_current_count())

        t1 = perf_counter()
        labels, distances = index.knn_query(emb, k=k_eff)
        dt = perf_counter() - t1

        retrieved.append(labels[0].tolist())
        times_sec.append(dt)

    return retrieved, times_sec


def main(
    val_pkl='validation_embeddings_spatial_only.pkl',
    db_pkl='database_embeddings_spatial_only.pkl',
    train_json='3.5_refined_train0824.json',
    output_json='1-retrieval_results_top1_spatial.json',
    topk=1,
    n_clusters=10,
    ef_construction=200,
    M=16,
    ef_search=50
):
    val_ids, val_embs = load_pickle_embeddings(val_pkl)
    db_ids,  db_embs  = load_pickle_embeddings(db_pkl)

    with open(train_json, 'r', encoding='utf-8') as f:
        train_entries = json.load(f)

    dim = db_embs.shape[1]
    kmeans, hnsw_indices = build_hnsw_indices(
        db_embs, n_clusters=n_clusters, dim=dim,
        ef_construction=ef_construction, M=M, ef_search=ef_search
    )

    retrieved_indices, times_sec = retrieve_topk(val_embs, kmeans, hnsw_indices, topk=topk)

    total = len(times_sec)
    for i, (vid, t) in enumerate(zip(val_ids, times_sec), start=1):
        print(f"[{i}/{total}] pure-retrieval (knn_query) for '{vid}': {t * 1000:.3f} ms")
    avg_ms = (sum(times_sec) / max(1, total)) * 1000.0
    print(f"Average pure-retrieval time over {total} queries: {avg_ms:.3f} ms")

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