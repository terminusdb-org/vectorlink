# Abt-Buy Evaluation Methodology

How to think about the Abt-Buy entity-resolution benchmark, its scoring, and what
100% F1 means operationally.

---

## 1. What the Abt-Buy task actually is

Abt-Buy is a **binary entity resolution** task between two product tables:

- Left table: 1,081 products from Abt.com.
- Right table: 1,092 products from Buy.com.
- Gold standard: 1,097 pairs of records declared to be the *same* product (the
  "perfect mapping").

Why 1,097 pairs for 1,081 vs 1,092?

- You are not matching 1:1 by index; you are matching by *semantics* (same
  real-world product).
- Some products in one catalogue have **no** counterpart in the other.
- Some products might have **multiple variants/duplicate entries** that
  legitimately map to the same counterpart, or near-duplicates that the gold
  creators decided to treat as "same".
- The gold is simply a set G of 1,097 record pairs (a_i, b_j) where a_i is from
  Abt and b_j is from Buy. There is no requirement that every row is used exactly
  once.

So think of Abt-Buy as:

> Universe of possible pairs U = A x B (1,081 x 1,092 ~ 1.18M).
> Gold positive pairs G is a subset of U with size 1,097.

Your system outputs some set of predicted positive pairs P subset of U. F1 = 100%
is a statement about the relationship between P and G.

---

## 2. Precision, recall, F1 in set terms

Given:

- Gold positives G (size 1,097).
- Your predicted positives P.

Define:

- True positives (TP): TP = |P intersection G|.
- False positives (FP): FP = |P \ G|.
- False negatives (FN): FN = |G \ P|.

Then:

- Precision = TP / (TP + FP).
- Recall = TP / (TP + FN).
- F1 = 2 * (precision * recall) / (precision + recall).

To get **F1 = 1.0** you must have:

- P = G.
- i.e., every predicted pair is correct (no FP) and every gold pair is predicted
  (no FN).
- Concretely:
  - TP = 1,097.
  - FP = 0.
  - FN = 0.
  - So precision = 1, recall = 1, F1 = 1.

Any deviation — missing a single gold pair or hallucinating a single non-gold
pair — drops F1 below 1.

---

## 3. The conceptual ER pipeline for Abt-Buy

Here is a generic, tool-agnostic pipeline that, *if perfect at each step*, yields
100% F1.

### 3.1. Ingestion and normalisation

Objective: get both tables into a clean, comparable representation.

Steps:

- Load the two CSVs:
  - `Abt.csv` (attributes: id, name, description, price).
  - `Buy.csv` (attributes: id, name, description, manufacturer, price).
- Normalise text fields per record:
  - Lowercase.
  - Strip punctuation, HTML, extra whitespace.
  - Normalise units and common abbreviations (e.g., "GB" vs "GByte", "inches"
    vs `"`).
- Normalise numeric fields:
  - Parse prices as numeric, standardise currency if needed (Abt-Buy is
    single-currency, so simple).
- Optional but helpful: derive extra features:
  - Tokenised product name.
  - Extract key attributes: screen size, capacity, model numbers using regexes.
  - Brand/manufacturer canonicalisation (e.g., "hewlett packard" -> "hp").

To reach 100% F1, your normalisation must be **information-preserving**: it cannot
destroy signals that the gold pairs rely on.

### 3.2. Blocking (candidate generation)

You cannot feasibly consider all ~1.18M pairs for large datasets, but for Abt-Buy
you *can* brute-force; still, reasoning in terms of blocking is useful.

Goal:

> Produce a set of candidate pairs C such that G subset of C.
> i.e., you never prune away a true match.

Strategies (conceptual):

- Soft blocking keys based on tokens:
  - For each Abt product, consider Buy products that share at least one rare
    token from the name (e.g., "HX2490b", "Pavilion", "ThinkPad").
- Brand + category blocking:
  - Match only within same normalised brand, and roughly similar category terms
    ("TV", "receiver", "notebook").

For 100% F1, this step must have **recall 1.0** at the candidate level:

- For every (a, b) in G, (a, b) must be present in C.
- You can overshoot; C may be large, but you cannot miss any gold pair.

In set terms: G subset of C subset of U.

### 3.3. Pairwise feature construction

For each candidate pair (a, b) in C, you build a feature vector capturing
similarity.

High-level feature types:

- String similarity:
  - Name similarity (e.g., token Jaccard, cosine over TF-IDF, character n-gram
    overlap).
  - Description similarity.
  - Brand equality / similarity.
- Numeric similarity:
  - Price difference.
  - Normalised price ratio.
- Structural/symbolic:
  - Exact or near-exact match of model numbers.
  - Same diagonal attributes (e.g., screen size, capacity).

For 100% F1, there must be **no gold pair whose feature vector is
indistinguishable from a non-match** in a way the classifier cannot separate.
Practically: your feature space must be expressive enough that every gold pair is
"more similar" than any plausible non-match.

---

## 4. The classifier and decision rule

### 4.1. Binary classifier

Conceptually, you train a binary classifier f(a, b) -> [0, 1] that outputs a
match probability (or score):

- Input: pairwise features.
- Output: score s representing "likelihood of same product".

It could be anything: logistic regression, gradient boosting, or a transformer
encoder applied to concatenated texts; but conceptually it is just a function
mapping features to a probability.

For **perfect F1 on the test split**, you need:

- For all gold pairs: s > t (above the threshold).
- For all non-gold pairs: s < t.

Or, more strongly: the classifier assigns **strictly higher scores to all gold
pairs than to any non-gold pairs**, so that some threshold t can cleanly separate
the two sets.

### 4.2. Threshold selection

Given classifier scores for all candidate pairs, you choose a threshold t such
that:

- P = {(a, b) in C : s(a, b) >= t}.

F1 is computed between P and G.

To get F1 = 1.0 you must be able to find a threshold where:

- Every gold pair's score >= t.
- Every non-gold pair's score < t.

Operationally, there are two variants:

- **Oracle threshold**: choose t that maximises F1 *using the gold labels*. This
  is what people often do in papers when tuning.
- **Realistic**: tune t on a validation split and hope it generalises. You still
  need the gold to cooperate.

But mathematically, if your classifier produces fully separable scores (all
positives strictly above all negatives), then such a threshold exists.

---

## 5. Evaluation: computing 100% F1 concretely

Assume you have:

- The gold file: set G of 1,097 pairs.
- Your predicted file: set P of pairs.

Evaluation script typically:

1. Reads G and P.
2. Computes TP, FP, FN by set operations on pairs.
3. Calculates precision, recall, F1 as above.

For 100% F1, the script must see:

- |P| = 1,097.
- Every pair in P appears in G.
- Every pair in G appears in P.

A minimal "perfect prediction" example:

- Gold contains pair (Abt row 123, Buy row 456).
- Your output file includes exactly that pair (same identifiers / indices, exact
  format expected by the script).
- Repeat for all 1,097 gold pairs and no others.

If you accidentally reorder columns, use the wrong ID field (e.g., line numbers
vs primary key), or output duplicates, the script may register FP or FN, reducing
F1.

---

## 6. End-to-end chain of operations, step by step

Putting it together, an ideal **conceptual** pipeline for Abt-Buy looks like:

1. **Load data and gold**
   - Load Abt and Buy tables.
   - Load gold mapping (1,097 pairs).
   - Ensure each row has a stable unique identifier.

2. **Preprocess / normalise**
   - Normalise text (lowercase, punctuation, whitespace).
   - Normalise brands, extract model numbers, parse prices.
   - Store both raw and normalised forms.

3. **Blocking (candidate generation)**
   - For each Abt record, create candidate Buy records using soft keys (shared
     tokens, brand, category).
   - Guarantee that all 1,097 gold pairs are included in your candidacy.
   - Optional: validate blocking recall by checking how many gold pairs are
     covered by C; if <100%, relax blocking.

4. **Feature engineering for candidates**
   - For each candidate pair, compute similarity features: token overlap, TF-IDF
     similarity, numeric distance, brand equality, etc.
   - Optionally, build a textual representation for a neural model (e.g.,
     concatenated title + description).

5. **Training data construction**
   - Label pairs:
     - Positives = all gold pairs in G.
     - Negatives = sample from C \ G (careful: some unlabelled pairs *might*
       actually match but are not in gold; Abt-Buy is curated though).
   - Split into train / validation / test at pair or entity level.

6. **Train classifier**
   - Train a binary classifier f(a, b) on features to separate positives from
     negatives.
   - Use validation set to tune hyperparameters.

7. **Score all candidate pairs in test**
   - For each candidate pair in test, compute s = f(a, b).

8. **Choose threshold t**
   - On validation: sweep thresholds, compute precision, recall, F1; pick t*
     that maximises F1.
   - Apply t* to test scores; predictions P are pairs with score >= t*.

9. **Evaluate on test**
   - Compute TP, FP, FN, precision, recall, F1 between P and gold G_test.
   - You get F1 = 1.0 if and only if P = G_test.

10. **Error analysis (if not 100%)**
    - False negatives: gold pairs your system missed -> check blocking (were they
      candidates?), features (did model numbers differ?), threshold (too high?).
    - False positives: predicted pairs not in gold -> inspect whether they are
      "reasonable" matches that gold did not include, or genuine errors in your
      similarity logic.

In an abstract sense, a **perfect pipeline** is just:

> A feature mapping and classifier such that the induced decision boundary
> perfectly separates the 1,097 positive pairs from all other candidate pairs,
> *plus* a blocking strategy that never drops a true pair, *plus* a threshold
> that realises that separation.

---

## 7. What "using Abt-Buy as a toy" should teach you

Using Abt-Buy as a conceptual toy for ER gives you:

- A clear mental model: ER as **set prediction over pairs**, evaluated with
  precision/recall/F1 on a labelled subset G.
- A concrete understanding that:
  - Blocking is about **pair search space recall**.
  - Classifier + features are about **pair classification precision/recall**.
  - Thresholding is about the **precision/recall trade-off**.
- A crisp condition: "100% F1" means "I reproduced the gold set exactly, not
  more, not less".

Once you internalise that, you can swap in any domain (companies, people,
financial instruments) and design analogous steps: clean, block, featurise,
classify, threshold, evaluate.

---

## 8. Implications for our vector-native framework (spec 17)

Our framework maps onto this standard ER pipeline as follows:

| ER Pipeline Step | Our Framework | Notes |
|-----------------|---------------|-------|
| Normalisation | Rendering templates (§3) | The domain-specific surface |
| Blocking | Reciprocal cross k-NN (§4.2) | Scoped by doc_type |
| Feature construction | Vector distance | Single feature: cosine distance |
| Classification | Distance threshold tau | tau = 0.5 by default |
| Decision rule | Mutual top-K grounding + assignment | §4.4 + §4.5 |

### 8.1. Scope: embedding-only resolution

Our framework is deliberately limited to what **text-based embedding similarity**
can achieve. We do not incorporate learned classifiers, symbolic rules, feature
engineering over structured fields, or any training on gold labels. The resolver
uses a single signal — cosine distance over embedded text — and deterministic
post-processing (grounding + assignment).

This means **F1 will not be 1.0**. That is expected and acceptable. The purpose
of the benchmark is to measure how well the embedding-based resolver performs as
a **first-pass matcher**, giving the user a high-quality candidate set that they
can refine with additional techniques (rule-based filters, human review, learned
re-rankers, etc.) to push precision higher.

### 8.2. What the scorer must measure

The critical constraint from the standard evaluation methodology:

> **P must equal G for F1 = 1.0. Every extra predicted pair is a false positive.
> Every missing gold pair is a false negative.**

The scorer must evaluate the **final prediction set P** — the set of (abtId,
buyId) pairs the system declares as matches — against the gold set G of 1,097
pairs. Precision and recall must both be **relevant and meaningful**:

- **Precision** answers: "of the pairs the resolver commits to, how many are
  correct?" This must not be deflated by algorithmic artefacts (e.g., emitting
  multiple Buy per Abt from intermediate grounding steps that the system does not
  actually commit to as predictions).
- **Recall** answers: "of the gold pairs, how many did the resolver find?" This
  must use the full gold pair count (1,097) as denominator.
- **F1** is the harmonic mean — it must reflect the genuine precision/recall
  trade-off of the embedding-based system, not an artificial distortion.

The prediction set P is what the system **declares as its answer**. It is not the
raw intermediate output of the algorithm (which may contain multiple candidates
per source record for internal processing reasons). P is the final, committed set
of resolved pairs — one prediction per source record at most (since the truth is
effectively one-to-one from Abt).

### 8.3. Comparability across modes

All three retrieval modes (search, similar, duplicates) feed the same resolver
and must be scored on the same basis. The headline F1 must be comparable:

- A mode that retrieves k > 1 candidates per record must not be penalised for
  having richer intermediate data — only its FINAL committed pairs count.
- A mode that retrieves k = 1 candidates must not appear artificially better
  simply because it has fewer intermediate artefacts.
- The comparison must be: "given the same resolve-and-commit step, which mode's
  embeddings produce the best final P?"

This means the scorer evaluates the resolver's **committed output** (one pair per
source record), not the raw grounding/assignment intermediate state.

---

## References

1. Benchmark datasets for entity resolution — Database Group Leipzig.
   https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution
2. Abt-Buy — OpenDataLab.
   https://opendatalab.com/OpenDataLab/Abt-Buy
3. A Deep Dive Into Cross-Dataset Entity Matching with Large and Small Language
   Models (EDBT 2025).
   https://openproceedings.org/2025/conf/edbt/paper-224.pdf
4. Scalable Entity Resolution for Web Product Descriptions (Information Fusion
   2020).
   https://personal.eur.nl/frasincar/papers/InfFus2020/inffus2020.pdf
5. A Machine Learning Approach for Product Matching and Categorisation (Semantic
   Web Journal).
   https://www.semantic-web-journal.net/system/files/swj1664.pdf
6. Heterogeneity in Entity Matching: A Survey and Experimental Analysis.
   https://arxiv.org/html/2508.08076v1
7. Online Entity Resolution Using an Oracle (VLDB 2016).
   http://www.vldb.org/pvldb/vol9/p384-firmani.pdf
8. We Put Our Matching Engine Against the Industry's Toughest Public Benchmarks
   — ListMatchGenie.
   https://listmatchgenie.com/blog/data-matching-public-benchmarks-results
