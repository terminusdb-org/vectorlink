#![forbid(unsafe_code)]

//! Entity resolution matching core (Spec 17 section 4).
//!
//! PURE — consumes the retrieval results (top-K neighbour lists for both cross
//! directions) and produces resolved pairs with a 3-partition output. Performs NO
//! I/O. All side-effecting retrieval lives in the service/store layers.
//!
//! Input shape:
//!   set_to_target: HashMap<set_id, Vec<Neighbour { id: target_id, distance }>>
//!   target_to_set: HashMap<target_id, Vec<Neighbour { id: set_id, distance }>>
//! Distances are on the reference [0,1] cosine scale (0 identical, 0.5 orthogonal).
//!
//! THREE INDEPENDENT THRESHOLDS:
//!   tau_one_to_one   — closeness for the 1:1 mutual-best CORE (reciprocal pairs)
//!   tau_one_to_many  — closeness for ADDITIONAL set-side matches
//!   tau_many_to_one  — closeness for ADDITIONAL target-side matches
//!
//! Output: ResolveResult { matched, set_only, target_only, stats }

use std::collections::{HashMap, HashSet};

use serde::Serialize;

// ─────────────────────────── Public types ────────────────────────────────────

/// A single neighbour in a top-K list.
#[derive(Debug, Clone)]
pub struct Neighbour {
    pub id: String,
    pub distance: f32,
}

/// A resolved match between a set record and a target record.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResolvedMatch {
    pub set_id: String,
    pub target_id: String,
    pub distance: f32,
    pub stage: MatchStage,
}

/// Which resolution stage produced a match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchStage {
    Core,
    SetExtra,
    TargetExtra,
}

/// Resolution statistics.
#[derive(Debug, Clone, Serialize)]
pub struct ResolveStats {
    pub k: usize,
    pub threshold: f32,
    pub tau_one_to_one: f32,
    pub tau_one_to_many: Option<f32>,
    pub tau_many_to_one: Option<f32>,
    pub edge_count: usize,
    pub core_count: usize,
    pub set_extra_count: usize,
    pub target_extra_count: usize,
    pub matched_count: usize,
    pub set_only_count: usize,
    pub target_only_count: usize,
    pub set_points: usize,
    pub target_points: usize,
    pub elapsed_ms: u64,
}

/// The full 3-partition resolve output.
#[derive(Debug, Clone, Serialize)]
pub struct ResolveResult {
    pub matched: Vec<ResolvedMatch>,
    pub set_only: Vec<String>,
    pub target_only: Vec<String>,
    pub stats: ResolveStats,
}

/// Options for the resolve algorithm.
#[derive(Debug, Clone)]
pub struct ResolveOptions {
    pub k: usize, // how many nearest neighbors to return
    pub threshold: f32,
    pub tau_one_to_one: f32, // threshold for one-to-one matches
    pub tau_one_to_many: Option<f32>, // threshold for one-to-many matches
    pub tau_many_to_one: Option<f32>, // threshold for many-to-one matches
}

// ─────────────────────────── Internal types ──────────────────────────────────

/// A candidate edge in the bipartite graph.
#[derive(Debug, Clone)]
struct Edge {
    set_id: String,
    target_id: String,
    distance: f32,
}

/// The sparse bipartite candidate graph.
struct CandidateGraph {
    /// All edges (keyed by "set_id::target_id" for dedup).
    edges: HashMap<String, Edge>,
    /// For each set id, the target ids in its top-K.
    set_top_k: HashMap<String, HashSet<String>>,
    /// For each target id, the set ids in its top-K.
    target_top_k: HashMap<String, HashSet<String>>,
}

// ─────────────────────────── Algorithm ───────────────────────────────────────

/// Run the full entity resolution algorithm.
///
/// PURE FUNCTION — no I/O. The elapsed_ms in stats must be set by the caller.
pub fn resolve(
    set_to_target: &HashMap<String, Vec<Neighbour>>,
    target_to_set: &HashMap<String, Vec<Neighbour>>,
    options: &ResolveOptions,
    elapsed_ms: u64,
) -> ResolveResult {
    // The candidate graph uses the loosest active tau as its edge cap: we need
    // all edges that ANY of the three thresholds might admit.
    let graph_tau = max_active_tau(options);
    let graph = build_candidate_graph(set_to_target, target_to_set, options.k, graph_tau);

    // Step 1: Core (1:1 mutual-best, reciprocal pairs passing tau_one_to_one).
    let core_pairs = ground_core(&graph, options.tau_one_to_one);

    // Step 2: Set-side extras (additional targets per set record).
    let set_extra_pairs = set_extras(&graph, &core_pairs, options.tau_one_to_many);

    // Step 3: Target-side extras (additional set records per target).
    let target_extra_pairs = target_extras(&graph, &core_pairs, options.tau_many_to_one);

    // Deduplicate: a pair may appear in both set_extra and target_extra.
    // Priority: core > set_extra > target_extra. Ties: keep min distance.
    let matched = dedup_matches(core_pairs, set_extra_pairs, target_extra_pairs);

    // 3-partition output.
    let matched_set_ids: HashSet<&str> = matched.iter().map(|m| m.set_id.as_str()).collect();
    let matched_target_ids: HashSet<&str> =
        matched.iter().map(|m| m.target_id.as_str()).collect();

    let set_only: Vec<String> = set_to_target
        .keys()
        .filter(|id| !matched_set_ids.contains(id.as_str()))
        .cloned()
        .collect();

    let target_only: Vec<String> = target_to_set
        .keys()
        .filter(|id| !matched_target_ids.contains(id.as_str()))
        .cloned()
        .collect();

    let core_count = matched.iter().filter(|m| m.stage == MatchStage::Core).count();
    let set_extra_count = matched
        .iter()
        .filter(|m| m.stage == MatchStage::SetExtra)
        .count();
    let target_extra_count = matched
        .iter()
        .filter(|m| m.stage == MatchStage::TargetExtra)
        .count();

    ResolveResult {
        stats: ResolveStats {
            k: options.k,
            threshold: options.threshold,
            tau_one_to_one: options.tau_one_to_one,
            tau_one_to_many: options.tau_one_to_many,
            tau_many_to_one: options.tau_many_to_one,
            edge_count: graph.edges.len(),
            core_count,
            set_extra_count,
            target_extra_count,
            matched_count: matched.len(),
            set_only_count: set_only.len(),
            target_only_count: target_only.len(),
            set_points: set_to_target.len(),
            target_points: target_to_set.len(),
            elapsed_ms,
        },
        matched,
        set_only,
        target_only,
    }
}

/// The loosest active tau (used as the graph edge cap).
fn max_active_tau(options: &ResolveOptions) -> f32 {
    let mut max = options.tau_one_to_one;
    if let Some(tau) = options.tau_one_to_many {
        if tau > max {
            max = tau;
        }
    }
    if let Some(tau) = options.tau_many_to_one {
        if tau > max {
            max = tau;
        }
    }
    max
}

/// Build the sparse bipartite candidate graph from the two directional top-K
/// lists, capped at `graph_tau`.
fn build_candidate_graph(
    set_to_target: &HashMap<String, Vec<Neighbour>>,
    target_to_set: &HashMap<String, Vec<Neighbour>>,
    k: usize,
    graph_tau: f32,
) -> CandidateGraph {
    let mut edges: HashMap<String, Edge> = HashMap::new();
    let mut set_top_k: HashMap<String, HashSet<String>> = HashMap::new();
    let mut target_top_k: HashMap<String, HashSet<String>> = HashMap::new();

    // From set→target direction.
    for (set_id, neighbours) in set_to_target {
        let top_k_slice = &neighbours[..neighbours.len().min(k)];
        let members: HashSet<String> = top_k_slice.iter().map(|n| n.id.clone()).collect();
        set_top_k.insert(set_id.clone(), members);

        for n in top_k_slice {
            if n.distance > graph_tau {
                continue;
            }
            let key = edge_key(set_id, &n.id);
            let entry = edges.entry(key).or_insert_with(|| Edge {
                set_id: set_id.clone(),
                target_id: n.id.clone(),
                distance: n.distance,
            });
            if n.distance < entry.distance {
                entry.distance = n.distance;
            }
        }
    }

    // From target→set direction.
    for (target_id, neighbours) in target_to_set {
        let top_k_slice = &neighbours[..neighbours.len().min(k)];
        let members: HashSet<String> = top_k_slice.iter().map(|n| n.id.clone()).collect();
        target_top_k.insert(target_id.clone(), members);

        for n in top_k_slice {
            if n.distance > graph_tau {
                continue;
            }
            // The set id here is n.id (the target→set lookup returns set ids).
            let key = edge_key(&n.id, target_id);
            let entry = edges.entry(key).or_insert_with(|| Edge {
                set_id: n.id.clone(),
                target_id: target_id.clone(),
                distance: n.distance,
            });
            if n.distance < entry.distance {
                entry.distance = n.distance;
            }
        }
    }

    CandidateGraph {
        edges,
        set_top_k,
        target_top_k,
    }
}

fn edge_key(set_id: &str, target_id: &str) -> String {
    format!("{}::{}", set_id, target_id)
}

/// CORE: mutual top-K grounding (tau_one_to_one).
/// For each set record, the NEAREST mutual edge <= tau is grounded.
/// A target may be the core match for several set records (target reusable in core).
fn ground_core(graph: &CandidateGraph, tau_one_to_one: f32) -> Vec<ResolvedMatch> {
    // Collect the NEAREST mutual edge per set record that passes tau_one_to_one.
    let mut best_by_set: HashMap<&str, &Edge> = HashMap::new();

    for edge in graph.edges.values() {
        if edge.distance > tau_one_to_one {
            continue;
        }
        // Check mutual membership: target in set's top-K AND set in target's top-K.
        let target_in_set_top_k = graph
            .set_top_k
            .get(&edge.set_id)
            .is_some_and(|s| s.contains(&edge.target_id));
        let set_in_target_top_k = graph
            .target_top_k
            .get(&edge.target_id)
            .is_some_and(|s| s.contains(&edge.set_id));

        if target_in_set_top_k && set_in_target_top_k {
            let current = best_by_set.get(edge.set_id.as_str());
            let replace = match current {
                None => true,
                Some(existing) => {
                    edge.distance < existing.distance
                        || (edge.distance == existing.distance
                            && edge.target_id < existing.target_id)
                }
            };
            if replace {
                best_by_set.insert(&edge.set_id, edge);
            }
        }
    }

    best_by_set
        .values()
        .map(|edge| ResolvedMatch {
            set_id: edge.set_id.clone(),
            target_id: edge.target_id.clone(),
            distance: edge.distance,
            stage: MatchStage::Core,
        })
        .collect()
}

/// SET-SIDE EXTRAS (tau_one_to_many): additional targets per set record.
/// Directional constraint: the SET record must have this target in its top-K.
fn set_extras(
    graph: &CandidateGraph,
    core_pairs: &[ResolvedMatch],
    tau_one_to_many: Option<f32>,
) -> Vec<ResolvedMatch> {
    let tau = match tau_one_to_many {
        Some(t) => t,
        None => return Vec::new(),
    };

    let core_keys: HashSet<String> = core_pairs
        .iter()
        .map(|p| edge_key(&p.set_id, &p.target_id))
        .collect();

    let mut extras = Vec::new();

    for edge in graph.edges.values() {
        if edge.distance > tau {
            continue;
        }
        let key = edge_key(&edge.set_id, &edge.target_id);
        if core_keys.contains(&key) {
            continue;
        }
        // Directional: set record must have this target in its top-K.
        let in_set_top_k = graph
            .set_top_k
            .get(&edge.set_id)
            .is_some_and(|s| s.contains(&edge.target_id));
        if !in_set_top_k {
            continue;
        }
        extras.push(ResolvedMatch {
            set_id: edge.set_id.clone(),
            target_id: edge.target_id.clone(),
            distance: edge.distance,
            stage: MatchStage::SetExtra,
        });
    }

    extras
}

/// TARGET-SIDE EXTRAS (tau_many_to_one): additional set records per target.
/// Directional constraint: the TARGET must have this set record in its top-K.
fn target_extras(
    graph: &CandidateGraph,
    core_pairs: &[ResolvedMatch],
    tau_many_to_one: Option<f32>,
) -> Vec<ResolvedMatch> {
    let tau = match tau_many_to_one {
        Some(t) => t,
        None => return Vec::new(),
    };

    let core_keys: HashSet<String> = core_pairs
        .iter()
        .map(|p| edge_key(&p.set_id, &p.target_id))
        .collect();

    let mut extras = Vec::new();

    for edge in graph.edges.values() {
        if edge.distance > tau {
            continue;
        }
        let key = edge_key(&edge.set_id, &edge.target_id);
        if core_keys.contains(&key) {
            continue;
        }
        // Directional: target must have this set record in its top-K.
        let in_target_top_k = graph
            .target_top_k
            .get(&edge.target_id)
            .is_some_and(|s| s.contains(&edge.set_id));
        if !in_target_top_k {
            continue;
        }
        extras.push(ResolvedMatch {
            set_id: edge.set_id.clone(),
            target_id: edge.target_id.clone(),
            distance: edge.distance,
            stage: MatchStage::TargetExtra,
        });
    }

    extras
}

/// Deduplicate matches across stages. A pair appearing in multiple stages keeps
/// the highest-priority version: core > set_extra > target_extra.
/// Within the same stage, keep the minimum distance.
fn dedup_matches(
    core: Vec<ResolvedMatch>,
    set_extra: Vec<ResolvedMatch>,
    target_extra: Vec<ResolvedMatch>,
) -> Vec<ResolvedMatch> {
    let mut map: HashMap<String, ResolvedMatch> = HashMap::new();

    let stage_rank = |s: MatchStage| -> u8 {
        match s {
            MatchStage::Core => 0,
            MatchStage::SetExtra => 1,
            MatchStage::TargetExtra => 2,
        }
    };

    let add = |map: &mut HashMap<String, ResolvedMatch>, pair: ResolvedMatch| {
        let key = edge_key(&pair.set_id, &pair.target_id);
        match map.get(&key) {
            None => {
                map.insert(key, pair);
            }
            Some(existing) => {
                let existing_rank = stage_rank(existing.stage);
                let new_rank = stage_rank(pair.stage);
                if new_rank < existing_rank
                    || (new_rank == existing_rank && pair.distance < existing.distance)
                {
                    map.insert(key, pair);
                }
            }
        }
    };

    for p in core {
        add(&mut map, p);
    }
    for p in set_extra {
        add(&mut map, p);
    }
    for p in target_extra {
        add(&mut map, p);
    }

    map.into_values().collect()
}

// ─────────────────────────── Tests ───────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a neighbour list from tuples.
    fn neighbours(items: &[(&str, f32)]) -> Vec<Neighbour> {
        items
            .iter()
            .map(|(id, dist)| Neighbour {
                id: id.to_string(),
                distance: *dist,
            })
            .collect()
    }

    #[test]
    fn mutual_best_forms_core() {
        // A→B at 0.1, B→A at 0.1 — mutual, should ground.
        let mut set_to_target: HashMap<String, Vec<Neighbour>> = HashMap::new();
        set_to_target.insert("A".to_owned(), neighbours(&[("B", 0.1)]));

        let mut target_to_set: HashMap<String, Vec<Neighbour>> = HashMap::new();
        target_to_set.insert("B".to_owned(), neighbours(&[("A", 0.1)]));

        let opts = ResolveOptions {
            k: 5,
            threshold: 0.5,
            tau_one_to_one: 0.45,
            tau_one_to_many: None,
            tau_many_to_one: None,
        };

        let result = resolve(&set_to_target, &target_to_set, &opts, 0);
        assert_eq!(result.matched.len(), 1);
        assert_eq!(result.matched[0].set_id, "A");
        assert_eq!(result.matched[0].target_id, "B");
        assert_eq!(result.matched[0].stage, MatchStage::Core);
        assert!(result.set_only.is_empty());
        assert!(result.target_only.is_empty());
    }

    #[test]
    fn non_mutual_does_not_ground() {
        // A→B at 0.1, but B→C at 0.05 (B's best is C, not A).
        let mut set_to_target: HashMap<String, Vec<Neighbour>> = HashMap::new();
        set_to_target.insert("A".to_owned(), neighbours(&[("B", 0.1)]));

        let mut target_to_set: HashMap<String, Vec<Neighbour>> = HashMap::new();
        target_to_set.insert("B".to_owned(), neighbours(&[("C", 0.05)]));

        let opts = ResolveOptions {
            k: 5,
            threshold: 0.5,
            tau_one_to_one: 0.45,
            tau_one_to_many: None,
            tau_many_to_one: None,
        };

        let result = resolve(&set_to_target, &target_to_set, &opts, 0);
        assert!(result.matched.is_empty());
        assert_eq!(result.set_only.len(), 1);
        assert!(result.set_only.contains(&"A".to_owned()));
    }

    #[test]
    fn tau_one_to_one_filters_distant_mutual() {
        // A→B mutual at 0.5 — but tau_one_to_one is 0.3, so it should NOT ground.
        let mut set_to_target: HashMap<String, Vec<Neighbour>> = HashMap::new();
        set_to_target.insert("A".to_owned(), neighbours(&[("B", 0.5)]));

        let mut target_to_set: HashMap<String, Vec<Neighbour>> = HashMap::new();
        target_to_set.insert("B".to_owned(), neighbours(&[("A", 0.5)]));

        let opts = ResolveOptions {
            k: 5,
            threshold: 0.6,
            tau_one_to_one: 0.3,
            tau_one_to_many: None,
            tau_many_to_one: None,
        };

        let result = resolve(&set_to_target, &target_to_set, &opts, 0);
        assert!(result.matched.is_empty());
    }

    #[test]
    fn set_extras_independent_of_target_extras() {
        // A→B mutual at 0.1 (core). D→B mutual at 0.12 (also core — D has B in
        // set_top_K AND B has D in target_top_K, both <= tau_one_to_one).
        // A→C at 0.15 (set-side extra: not in core because C does not have A as
        // its NEAREST — but A has C in set_top_K).
        //
        // Use tau_one_to_one = 0.1 to EXCLUDE D→B from core (0.12 > 0.1), forcing
        // it into set_extra/target_extra territory. This isolates the directional
        // extra behaviour.
        let mut set_to_target: HashMap<String, Vec<Neighbour>> = HashMap::new();
        set_to_target.insert("A".to_owned(), neighbours(&[("B", 0.1), ("C", 0.15)]));
        set_to_target.insert("D".to_owned(), neighbours(&[("B", 0.12)]));

        let mut target_to_set: HashMap<String, Vec<Neighbour>> = HashMap::new();
        target_to_set.insert("B".to_owned(), neighbours(&[("A", 0.1), ("D", 0.12)]));
        target_to_set.insert("C".to_owned(), neighbours(&[("A", 0.15)]));

        let opts = ResolveOptions {
            k: 5,
            threshold: 0.5,
            tau_one_to_one: 0.1,    // Only A↔B at exactly 0.1 qualifies for core.
            tau_one_to_many: Some(0.2),
            tau_many_to_one: Some(0.2),
        };

        let result = resolve(&set_to_target, &target_to_set, &opts, 0);

        // Core: only A↔B (at distance exactly 0.1 = tau_one_to_one).
        let core: Vec<&ResolvedMatch> = result
            .matched
            .iter()
            .filter(|m| m.stage == MatchStage::Core)
            .collect();
        assert_eq!(core.len(), 1);
        assert_eq!(core[0].set_id, "A");
        assert_eq!(core[0].target_id, "B");

        // Set extras: A→C (A has C in set_top_K, 0.15 <= 0.2) AND D→B (D has B
        // in set_top_K, 0.12 <= 0.2). D→B also qualifies as target_extra (B has
        // D in target_top_K) but dedup keeps set_extra (higher priority).
        let set_extras: Vec<&ResolvedMatch> = result
            .matched
            .iter()
            .filter(|m| m.stage == MatchStage::SetExtra)
            .collect();
        assert_eq!(set_extras.len(), 2);
        let set_extra_pairs: HashSet<(&str, &str)> =
            set_extras.iter().map(|m| (m.set_id.as_str(), m.target_id.as_str())).collect();
        assert!(set_extra_pairs.contains(&("A", "C")));
        assert!(set_extra_pairs.contains(&("D", "B")));

        // No target_extras remain after dedup (D→B promoted to set_extra; A→C is set_extra).
        let target_extras: Vec<&ResolvedMatch> = result
            .matched
            .iter()
            .filter(|m| m.stage == MatchStage::TargetExtra)
            .collect();
        assert_eq!(target_extras.len(), 0);

        // D is matched (set_extra), so NOT in set_only.
        assert!(!result.set_only.contains(&"D".to_owned()));
    }

    #[test]
    fn three_partition_complete() {
        // A→B mutual (core). C has no match. Target X has no match.
        let mut set_to_target: HashMap<String, Vec<Neighbour>> = HashMap::new();
        set_to_target.insert("A".to_owned(), neighbours(&[("B", 0.1)]));
        set_to_target.insert("C".to_owned(), neighbours(&[("B", 0.8)])); // too far

        let mut target_to_set: HashMap<String, Vec<Neighbour>> = HashMap::new();
        target_to_set.insert("B".to_owned(), neighbours(&[("A", 0.1)]));
        target_to_set.insert("X".to_owned(), neighbours(&[("C", 0.9)])); // too far

        let opts = ResolveOptions {
            k: 5,
            threshold: 0.5,
            tau_one_to_one: 0.45,
            tau_one_to_many: None,
            tau_many_to_one: None,
        };

        let result = resolve(&set_to_target, &target_to_set, &opts, 0);
        assert_eq!(result.matched.len(), 1);
        assert!(result.set_only.contains(&"C".to_owned()));
        assert!(result.target_only.contains(&"X".to_owned()));
    }

    #[test]
    fn pure_target_extra_when_not_in_set_top_k() {
        // A→B mutual (core). Edge D→B exists only from B's direction (B has D in
        // target_top_K but D does NOT have B in set_top_K — D's set_to_target
        // points elsewhere). So D→B is purely a target_extra.
        let mut set_to_target: HashMap<String, Vec<Neighbour>> = HashMap::new();
        set_to_target.insert("A".to_owned(), neighbours(&[("B", 0.1)]));
        // D looks toward X (not B), so D's set_top_K does NOT contain B.
        set_to_target.insert("D".to_owned(), neighbours(&[("X", 0.9)]));

        let mut target_to_set: HashMap<String, Vec<Neighbour>> = HashMap::new();
        target_to_set.insert("B".to_owned(), neighbours(&[("A", 0.1), ("D", 0.12)]));
        target_to_set.insert("X".to_owned(), neighbours(&[("D", 0.9)]));

        let opts = ResolveOptions {
            k: 5,
            threshold: 0.5,
            tau_one_to_one: 0.45,
            tau_one_to_many: Some(0.2),
            tau_many_to_one: Some(0.2),
        };

        let result = resolve(&set_to_target, &target_to_set, &opts, 0);

        // Core: A↔B.
        let core: Vec<&ResolvedMatch> = result
            .matched
            .iter()
            .filter(|m| m.stage == MatchStage::Core)
            .collect();
        assert_eq!(core.len(), 1);

        // D→B is ONLY a target_extra: B has D in target_top_K (yes) but D does
        // NOT have B in set_top_K (D's top-K is [X]).
        let target_extras: Vec<&ResolvedMatch> = result
            .matched
            .iter()
            .filter(|m| m.stage == MatchStage::TargetExtra)
            .collect();
        assert_eq!(target_extras.len(), 1);
        assert_eq!(target_extras[0].set_id, "D");
        assert_eq!(target_extras[0].target_id, "B");
    }

    #[test]
    fn disabled_extras_produce_no_matches() {
        // With tau_one_to_many = None and tau_many_to_one = None,
        // only core matches should appear.
        let mut set_to_target: HashMap<String, Vec<Neighbour>> = HashMap::new();
        set_to_target.insert("A".to_owned(), neighbours(&[("B", 0.1), ("C", 0.15)]));

        let mut target_to_set: HashMap<String, Vec<Neighbour>> = HashMap::new();
        target_to_set.insert("B".to_owned(), neighbours(&[("A", 0.1)]));
        target_to_set.insert("C".to_owned(), neighbours(&[("A", 0.15)]));

        let opts = ResolveOptions {
            k: 5,
            threshold: 0.5,
            tau_one_to_one: 0.45,
            tau_one_to_many: None,
            tau_many_to_one: None,
        };

        let result = resolve(&set_to_target, &target_to_set, &opts, 0);
        // Only A↔B should core-ground (mutual best). A↔C is also mutual but
        // ground_core picks the NEAREST per set id, so only A↔B.
        let core: Vec<&ResolvedMatch> = result
            .matched
            .iter()
            .filter(|m| m.stage == MatchStage::Core)
            .collect();
        assert_eq!(core.len(), 1);
        assert_eq!(core[0].target_id, "B");
        // No extras.
        assert!(result
            .matched
            .iter()
            .all(|m| m.stage == MatchStage::Core));
    }

    #[test]
    fn dedup_prefers_higher_stage() {
        // Same pair in both set_extra and target_extra — keep set_extra.
        let core = vec![];
        let set_extra = vec![ResolvedMatch {
            set_id: "A".to_owned(),
            target_id: "B".to_owned(),
            distance: 0.15,
            stage: MatchStage::SetExtra,
        }];
        let target_extra = vec![ResolvedMatch {
            set_id: "A".to_owned(),
            target_id: "B".to_owned(),
            distance: 0.12,
            stage: MatchStage::TargetExtra,
        }];

        let result = dedup_matches(core, set_extra, target_extra);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].stage, MatchStage::SetExtra);
    }

    #[test]
    fn max_active_tau_picks_loosest() {
        let opts = ResolveOptions {
            k: 5,
            threshold: 0.5,
            tau_one_to_one: 0.3,
            tau_one_to_many: Some(0.4),
            tau_many_to_one: Some(0.2),
        };
        assert!((max_active_tau(&opts) - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn max_active_tau_with_disabled() {
        let opts = ResolveOptions {
            k: 5,
            threshold: 0.5,
            tau_one_to_one: 0.3,
            tau_one_to_many: None,
            tau_many_to_one: None,
        };
        assert!((max_active_tau(&opts) - 0.3).abs() < f32::EPSILON);
    }
}
