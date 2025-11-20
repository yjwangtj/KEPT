import json
import pickle
import numpy as np
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


def brute_force_topk_fullsort(
    val_embs_norm: np.ndarray,
    train_embs_norm: np.ndarray,
    topk: int = 1,
):
    n_train = train_embs_norm.shape[0]
    k_eff = min(max(int(topk), 1), n_train)

    topk_indices, topk_scores, times_sec = [], [], []

    for q in val_embs_norm:
        t0 = perf_counter()

        sims = train_embs_norm @ q
        idx_sorted_desc = np.argsort(-sims)
        idx_topk = idx_sorted_desc[:k_eff]
        scores_topk = sims[idx_topk]

        dt = perf_counter() - t0
        times_sec.append(dt)
        topk_indices.append(idx_topk.astype(int).tolist())
        topk_scores.append(scores_topk.astype(np.float32).tolist())

    return topk_indices, topk_scores, times_sec


def main(
    val_pkl='validation_embeddings_refined0824.pkl',
    train_pkl='database_embeddings_refined0824.pkl',
    train_json='3.5_refined_train0824.json',
    output_json='retrieval_results_top5_bruteforce_fullsort_trainDB_0824.json',
    topk=5
):
    val_ids, val_embs = load_pickle_embeddings(val_pkl)
    train_ids, train_embs = load_pickle_embeddings(train_pkl)

    with open(train_json, 'r', encoding='utf-8') as f:
        train_entries = json.load(f)

    val_embs_norm = l2_normalize(val_embs.astype(np.float32))
    train_embs_norm = l2_normalize(train_embs.astype(np.float32))

    topk_indices, topk_scores, times_sec = brute_force_topk_fullsort(
        val_embs_norm=val_embs_norm,
        train_embs_norm=train_embs_norm,
        topk=topk
    )

    total = len(times_sec)
    for i, (vid, t) in enumerate(zip(val_ids, times_sec), start=1):
        print(f"[{i}/{total}] brute-force FULL-SORT (train DB) for '{vid}': {t * 1000:.3f} ms")
    avg_ms = (sum(times_sec) / max(1, total)) * 1000.0
    print(f"Average brute-force FULL-SORT (train DB) over {total} queries: {avg_ms:.3f} ms")

    results = []
    for vid, idxs, scores in zip(val_ids, topk_indices, topk_scores):
        scenes = [train_entries[idx] for idx in idxs] if len(idxs) > 0 else []
        results.append({
            'validation_id': vid,
            'retrieved_db_indices': idxs,
            'retrieved_scenes': scenes,
            'retrieved_scores': [float(s) for s in scores]
        })

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved brute-force FULL-SORT Top-{topk} results (train DB) for {len(results)} validation items to '{output_json}'")


if __name__ == '__main__':
    main()