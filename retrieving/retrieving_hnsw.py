import os
import json
import pickle
import numpy as np
import hnswlib
from sklearn.cluster import KMeans

def load_pickle_embeddings(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    ids = data['ids']
    embs = np.array(data['embeddings'])
    return ids, embs

def build_hnsw_indices(db_embs, n_clusters=10, dim=128, ef_construction=200, M=16, ef_search=50):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(db_embs)
    labels = kmeans.labels_
    indices = {}
    for cid in range(n_clusters):
        idxs = np.where(labels == cid)[0]
        if len(idxs) == 0:
            continue
        p = hnswlib.Index(space='cosine', dim=dim)
        p.init_index(max_elements=len(idxs), ef_construction=ef_construction, M=M)
        p.add_items(db_embs[idxs], idxs)
        p.set_ef(ef_search)
        indices[cid] = p
    return kmeans, indices

def retrieve_topk(val_embs, kmeans, hnsw_indices, topk=1):
    retrieved = []
    for emb in val_embs:
        cid = int(kmeans.predict(emb.reshape(1, -1))[0])
        index = hnsw_indices.get(cid, None)
        if index is None:
            retrieved.append([])
        else:
            labels, distances = index.knn_query(emb, k=topk)
            retrieved.append(labels[0].tolist())
    return retrieved

def main(
    val_pkl='validation_embeddings_refined.pkl',
    db_pkl='database_embeddings_refined.pkl',
    train_json='3.5_refined_train.json',
    output_json='retrieval_results_top1.json',
    topk=1,
    n_clusters=10
):
    val_ids, val_embs = load_pickle_embeddings(val_pkl)
    db_ids,  db_embs  = load_pickle_embeddings(db_pkl)

    with open(train_json, 'r') as f:
        train_entries = json.load(f)

    dim = db_embs.shape[1]
    kmeans, hnsw_indices = build_hnsw_indices(
        db_embs, n_clusters=n_clusters, dim=dim
    )

    retrieved_indices = retrieve_topk(val_embs, kmeans, hnsw_indices, topk=topk)

    results = []
    for vid, nbrs in zip(val_ids, retrieved_indices):
        trajs = []
        for db_idx in nbrs:
            trajs.append(json.loads(train_entries[db_idx]['trajectories']))
        results.append({
            'validation_id': vid,
            'retrieved_db_indices': nbrs,
            'retrieved_trajectories': trajs
        })

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved retrieval results for {len(results)} validation items to '{output_json}'")

if __name__ == '__main__':
    main()
