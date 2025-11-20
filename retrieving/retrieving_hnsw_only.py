import json
import pickle
import numpy as np
import hnswlib
from time import perf_counter


def load_pickle_embeddings(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    ids = data['ids']
    embs = np.array(data['embeddings'], dtype=np.float32)
    return ids, embs


def build_hnsw_index(
    db_embs: np.ndarray,
    space: str = "cosine",
    ef_construction: int = 200,
    M: int = 16,
    ef_search: int = 128
):
    num_elements, dim = db_embs.shape
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
    labels = np.arange(num_elements, dtype=np.int32)
    index.add_items(db_embs, labels)
    index.set_ef(max(ef_search, 1))
    return index


def retrieve_topk_hnsw(val_embs: np.ndarray, index: hnswlib.Index, topk: int = 1):
    retrieved_indices = []
    times_sec = []
    k = min(topk, index.get_current_count())
    for emb in val_embs:
        t0 = perf_counter()
        labels, _ = index.knn_query(emb, k=k)
        dt = perf_counter() - t0
        retrieved_indices.append(labels[0].tolist())
        times_sec.append(dt)
    return retrieved_indices, times_sec


def main(
    val_pkl='validation_embeddings_refined0824.pkl',
    db_pkl='database_embeddings_refined0824.pkl',
    train_json='3.5_refined_train0824.json',
    output_json='retrieval_results_top5_hnsw_only_0825.json',
    topk=5,
    ef_construction=200,
    M=16,
    ef_search=128,
    space='cosine'
):
    val_ids, val_embs = load_pickle_embeddings(val_pkl)
    db_ids,  db_embs  = load_pickle_embeddings(db_pkl)

    with open(train_json, 'r', encoding='utf-8') as f:
        train_entries = json.load(f)

    index = build_hnsw_index(
        db_embs=db_embs,
        space=space,
        ef_construction=ef_construction,
        M=M,
        ef_search=max(ef_search, topk)
    )

    retrieved_indices, times_sec = retrieve_topk_hnsw(
        val_embs=val_embs,
        index=index,
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