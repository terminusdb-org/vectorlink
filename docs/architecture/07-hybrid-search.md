# 07 — Hybrid Search: Fusing Vector and Full-Text Results

### Learning objectives
After this document you will be able to:
- **Explain** why vectorlink fuses ranks rather than scores, and what MaxP and RRF contribute.
- **Interpret** the `distance`, `rrf_score`, and per-leg `vector`/`fts` fields of a hybrid hit.
- **Predict** how a document's rank changes when it appears in one leg vs. both.

### Prerequisites
[04 — Search](./04-search.md) and [05 — Embeddings & determinism](./05-embeddings-and-determinism.md) (chunking).

---

## 1. The problem: two legs, incommensurate scores

For each hybrid query, vectorlink issues two independent ranked lists against the same Lance snapshot:

- **Vector leg.** ANN cosine search over chunk embeddings (`scan().nearest("embedding", q, k)`), producing `_distance` in Lance's $[0,2]$ range, normalised to the reference $[0,1]$ scale for display.
- **FTS leg.** BM25 over the `content` column's inverted index (`scan().full_text_search(...)`), producing `_score` — unbounded and corpus-dependent.

The tempting move is combining the lists by score — a weighted sum of normalised cosine and BM25. IR research has shown this is fragile since TREC: the score distributions differ in scale, shape, and corpus dependence, so any fixed normalisation is arbitrary [@fox1993combining]. A query where BM25 is informative and one where it is noise get the same weight either way.

## 2. Mechanism: rank fusion, not score fusion

The standard resolution is **Reciprocal Rank Fusion** [@cormack2009rrf]: discard raw scores entirely and combine *rank positions*. For a document $d$ retrieved by a set of legs $L$:

$$
\mathrm{score}(d) = \sum_{\ell \in L} \frac{1}{k + r_\ell(d)}, \qquad k = 60
$$

where $r_\ell(d)$ is $d$'s rank in leg $\ell$. A document present in both legs accrues two contributions; a document strong in only one leg is penalised gracefully rather than eliminated. The constant $k$ damps deep ranks — $k = 60$ is near-optimal across corpora and the choice is not critical, which is precisely the appeal: RRF never compares incomparable numbers [@cormack2009rrf]. The same scheme is used by Azure AI Search, Elasticsearch, Vespa, and LanceDB's own `RRFReranker`.

> **Provenance note:** LanceDB's reranker fuses *row-level* results — in its data model every row is a document. vectorlink's rows are chunks, so the upstream reranker cannot express the collapse in §3 even where it is reachable; the same Cormack formula is applied at document level here instead.

## 3. Mechanism: documents vs. chunks — collapse each leg first

vectorlink indexes *passages*, not documents: a long page yields many chunk rows, each a candidate in both legs. Fusing chunk-level lists fails twice over — a multi-chunk document occupies several rank slots per leg (crowding other documents out), and a document's evidence is diluted across its passages. The latter was the observed cause of `schema-reference-guide` (42 chunks) failing to rank for its own title.

Passage-retrieval theory offers three reductions from passages to documents: **MaxP**, SumP, and AvgP [@callan1994passage; @dai2019deeper]. MaxP assigns each document the position of its best passage; SumP accumulates passage evidence and AvgP averages it. MaxP is the robust default: SumP is length-biased, letting long documents earn rank from many mediocre passages — the dilution problem in reverse [@dai2019deeper]. Formally, each leg $\ell$ collapses to a document ranking in which document $d$ takes the rank position of its best chunk among the leg's *distinct* documents:

$$
r_\ell(d) = \operatorname{rankpos}\!\Bigl( \min_{c \in d}\, r_\ell^{chunk}(c) \Bigr)
$$

The pipeline, end to end:

1. Run vector and FTS legs in parallel (`tokio::join!`) on one snapshot — same data version, no skew.
2. Collapse each leg to a document ranking via MaxP.
3. Sum $1 / (60 + r_\ell(d) + 1)$ over the legs each document appears in.
4. Sort by fused score descending; document id breaks ties deterministically.
5. Report the winning chunk as representative (snippet, index, token offset); the vector leg's chunk is preferred when present because it carries a geometrically meaningful distance.

## 4. Scoring: what `distance` means in hybrid mode

Rank fusion discards magnitudes — why it is robust — but it also means no leg's raw distance can honestly represent the fused result (a rank-4 hit may carry a *smaller* cosine distance than the rank-1 hit; reporting it as "the distance" contradicted the ordering). The top-level `distance` is therefore derived from the fused score:

$$
d_{\mathrm{fused}}(d) = 1 - \frac{\mathrm{score}(d)}{\mathrm{score}_{\max}}, \qquad \mathrm{score}_{\max} = \sum_{\ell \in L} \frac{1}{k + 1}
$$

i.e. the fraction of *perfect fused evidence* achieved, inverted into the $[0,1]$ distance convention. With two legs, $\mathrm{score}_{\max} = 2/61$. Reading it:

- **0** — ranked first in *both* legs; the strongest possible hybrid match.
- **0.5** — ranked first in *only one* leg. Single-leg hits can never beat 0.5, so dual-leg corroboration is visually obvious.
- **→1** — barely surfaced in either leg.
- Result order and distance order agree by construction — the property a human scanning `distance` actually needs.

For transparency, every hybrid hit also carries `rrf_score` (the raw fused score) and per-leg `{rank, distance}` under `vector` and `fts` — the real cosine / BM25-derived distance of that leg's best chunk — so the evidence is never hidden. In vector-only or FTS-only mode, `distance` keeps its single-leg meaning and the leg fields are absent.

## 5. Worked example (admin/docs corpus)

252 TerminusDB documentation pages, chunked at ~430 tokens with overlap. Query: `schema reference guide`.

| rank | document | distance | vector leg | fts leg |
|---|---|---|---|---|
| 0 | schema-reference-guide | 0.008 | r1 (0.393) | r0 (0.115) |
| 1 | glossary | 0.047 | r2 (0.396) | r4 (0.120) |
| 2 | woql-class-reference-guide | 0.068 | r6 (0.412) | r3 (0.120) |
| 3 | woql-triple-slice | 0.500 | r0 (0.094) | — |
| 4 | xsd-data-types | 0.508 | — | r1 (0.117) |

`woql-triple-slice` owns the single best chunk in the corpus (cosine 0.094) yet lands fourth — it has no lexical corroboration. Under the previous chunk-level fusion that raw distance won the document outright; under doc-level MaxP+RRF, one outstanding passage cannot outrank consensus. Conversely `schema-reference-guide` wins decisively on combined evidence, where chunk-level RRF followed by a distance re-sort had buried it.

## Check your understanding

1. Why does RRF not need score normalisation, and what does it sacrifice in exchange?
2. A document holds rank 0 in the vector leg and is absent from the FTS leg. What is its top-level `distance`, and why can a dual-leg rank-5 document still outrank it?
3. Why does SumP over chunk scores reintroduce the dilution problem that MaxP avoids?
4. Where would you look in a hybrid hit to find the actual cosine distance of the best-matching chunk?

## References

```bibtex
@inproceedings{cormack2009rrf,
  author    = {Cormack, Gordon V. and Clarke, Charles L. A. and B{\"u}ttcher, Stefan},
  title     = {Reciprocal Rank Fusion Outperforms {C}ondorcet and Individual Rank Learning Methods},
  booktitle = {Proceedings of the 32nd International {ACM} {SIGIR} Conference on Research and Development in Information Retrieval},
  pages     = {758--759},
  publisher = {ACM},
  year      = {2009},
  doi       = {10.1145/1571941.1572114}
}

@inproceedings{callan1994passage,
  author    = {Callan, James P.},
  title     = {Passage-Level Evidence in Document Retrieval},
  booktitle = {Proceedings of the 17th Annual International {ACM}-{SIGIR} Conference on Research and Development in Information Retrieval},
  pages     = {302--310},
  publisher = {ACM/Springer},
  year      = {1994},
  isbn      = {3-540-19889-X}
}

@inproceedings{dai2019deeper,
  author    = {Dai, Zhuyun and Callan, Jamie},
  title     = {Deeper Text Understanding for {IR} with Contextual Neural Language Modeling},
  booktitle = {Proceedings of the 42nd International {ACM} {SIGIR} Conference on Research and Development in Information Retrieval},
  pages     = {985--988},
  publisher = {ACM},
  year      = {2019},
  doi       = {10.1145/3331184.3331303}
}

@inproceedings{fox1993combining,
  author    = {Fox, Edward A. and Shaw, Joseph A.},
  title     = {Combination of Multiple Searches},
  booktitle = {Proceedings of the Second Text {RE}trieval Conference ({TREC}-2)},
  series    = {{NIST} Special Publication 500-215},
  pages     = {243--252},
  publisher = {National Institute of Standards and Technology},
  year      = {1993}
}
```
