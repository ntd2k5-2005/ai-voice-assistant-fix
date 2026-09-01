"""Retrieval metrics with explicit binary recall and graded/binary nDCG."""
import math
def recall_at_k(retrieved, relevant, k):
    if k <= 0 or not relevant: return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))
def hit_rate_at_k(retrieved, relevant, k):
    if k <= 0 or not relevant: return 0.0
    return float(bool(set(retrieved[:k]) & set(relevant)))
def ndcg_at_k(retrieved, relevance, k=10):
    gains=[float(relevance.get(cid, 0)) for cid in retrieved[:k]]
    dcg=sum(g/math.log2(i+2) for i,g in enumerate(gains))
    ideal=sorted((float(v) for v in relevance.values()), reverse=True)[:k]
    idcg=sum(g/math.log2(i+2) for i,g in enumerate(ideal))
    return dcg/idcg if idcg else 0.0
def mrr(retrieved, relevant):
    relevant=set(relevant)
    for index, chunk_id in enumerate(retrieved, 1):
        if chunk_id in relevant: return 1.0/index
    return 0.0
def aggregate_metrics(rows, ks=(1,5,10,20)):
    denominator=len(rows) or 1
    result={f"recall@{k}": sum(recall_at_k(r["retrieved"], r["relevant"], k) for r in rows)/denominator for k in ks}
    result.update({f"hit_rate@{k}": sum(hit_rate_at_k(r["retrieved"], r["relevant"], k) for r in rows)/denominator for k in (5,10)})
    result.update({"mrr": sum(mrr(r["retrieved"], r["relevant"]) for r in rows)/denominator, "ndcg@10": sum(ndcg_at_k(r["retrieved"], r.get("relevance", {x:1 for x in r["relevant"]})) for r in rows)/denominator})
    return result
