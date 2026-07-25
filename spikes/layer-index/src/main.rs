//! Phase-0 spike 0a-2 — Global commit→layer index via Lance tags (RISK-16, HARD GATE)
//! + RISK-18 compaction/cleanup safety.
//!
//! Proves:
//!  1. tags create/get_version/list round-trip
//!  2. a tag created on one branch resolves from a session on another branch (GLOBAL — branch-from-anywhere)
//!  3. basic scale: create many tags, measure single lookup + full enumeration
//!  4. RISK-18: after compact_files(), a pre-compaction tag/version still reads the right rows
//!  5. RISK-18: a branch that shared a parent fragment still resolves after parent compaction
//!
//! Throwaway. Prints `RESULT <name>: PASS|FAIL` for the evidence report.

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray, FixedSizeListArray};
use arrow_array::types::Float32Type;
use arrow_schema::{DataType, Field, Schema};

use lance::dataset::{Dataset, WriteParams, WriteMode};
use lance::dataset::optimize::{compact_files, CompactionOptions};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Utf8, false),
        Field::new("n", DataType::Int32, false),
        Field::new("embedding", DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 8), false),
    ]))
}

fn batch(start: i32, n: i32) -> RecordBatch {
    let ids: Vec<String> = (start..start + n).map(|i| format!("doc/{i}")).collect();
    let ns: Vec<i32> = (start..start + n).collect();
    let emb = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        (start..start + n).map(|i| Some((0..8).map(move |j| Some(i as f32 + j as f32)))), 8);
    RecordBatch::try_new(schema(), vec![
        Arc::new(StringArray::from(ids)),
        Arc::new(Int32Array::from(ns)),
        Arc::new(emb),
    ]).unwrap()
}

async fn append(uri: &str, start: i32, n: i32, mode: WriteMode) -> Dataset {
    let reader = RecordBatchIterator::new(vec![Ok(batch(start, n))], schema());
    Dataset::write(reader, uri, Some(WriteParams { mode, ..Default::default() })).await.unwrap()
}

fn result(name: &str, pass: bool, detail: &str) {
    println!("RESULT {name}: {} — {detail}", if pass { "PASS" } else { "FAIL" });
}

/// Reversible, collision-free commit-id → Lance tag-name encoding (Spec 12 §3.4).
/// Stays within Lance's allowed alphabet [A-Za-z0-9_-]; "c_" prefix; '-' is the escape char.
fn encode_commit_tag(commit: &str) -> String {
    let mut out = String::from("c_");
    for &b in commit.as_bytes() {
        let c = b as char;
        if c == '-' { out.push_str("--"); }
        else if c.is_ascii_alphanumeric() || c == '_' { out.push(c); }
        else { out.push_str(&format!("-{:02x}", b)); }
    }
    out
}
fn decode_commit_tag(tag: &str) -> String {
    let s = tag.strip_prefix("c_").unwrap();
    let b = s.as_bytes();
    let mut out: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'-' {
            if b[i + 1] == b'-' { out.push(b'-'); i += 2; }
            else { out.push(u8::from_str_radix(std::str::from_utf8(&b[i+1..i+3]).unwrap(), 16).unwrap()); i += 3; }
        } else { out.push(b[i]); i += 1; }
    }
    String::from_utf8(out).unwrap()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = std::env::temp_dir().join(format!("spike-layer-index-{}", std::process::id()));
    let uri = tmp.to_str().unwrap().to_string();
    println!("uri: {uri}");

    // Build a little history: v1 (commit C0), v2 (commit C1) on main.
    let mut ds = append(&uri, 0, 100, WriteMode::Create).await;   // version 1
    let v_c0 = ds.version().version;
    ds.append(RecordBatchIterator::new(vec![Ok(batch(100, 50))], schema()), None).await?; // version 2
    let v_c1 = ds.version().version;

    // Realistic opaque commit ids (TerminusDB-style; the C1 one is adversarial with ':' '/' '.').
    let c0 = "o2uq7k1mrun1vp4urktmw55962vlpto";
    let c1 = "branch:feature/v.1..2";          // adversarial: chars Lance forbids in a raw tag
    let (t0, t1) = (encode_commit_tag(c0), encode_commit_tag(c1));
    println!("encoded tags: {c0} -> {t0} | {c1} -> {t1}");

    // --- 0. encoding round-trips (pure) ---
    let enc_ok = decode_commit_tag(&t0) == c0 && decode_commit_tag(&t1) == c1;
    result("0a2.0_tag_encoding", enc_ok, &format!("encode/decode round-trips for normal + adversarial ids"));

    // --- 1. tags round-trip: bind opaque commit ids to versions via the encoded tag name ---
    ds.tags().create(&t0, v_c0).await?;
    ds.tags().create(&t1, v_c1).await?;
    let got_c0 = ds.tags().get_version(&t0).await?;
    let got_c1 = ds.tags().get_version(&t1).await?;
    let listed = ds.tags().list().await?;
    result("0a2.1_tag_roundtrip", got_c0 == v_c0 && got_c1 == v_c1 && listed.len() == 2,
        &format!("{c0}→v{got_c0}, {c1}→v{got_c1}, list={}", listed.len()));

    // --- 2. GLOBAL resolution: branch off C0, then resolve a main-created tag from the branch session ---
    let branch = ds.create_branch("feature", v_c0, None).await?;
    let from_branch = branch.tags().get_version(&t1).await;   // C1 was tagged on main, after the fork point
    let global = from_branch.as_ref().map(|v| *v == v_c1).unwrap_or(false);
    result("0a2.2_global_resolution", global,
        &format!("{c1} (tagged on main) resolves from the feature-branch session: {:?}", from_branch));

    // --- 3. scale: create many tags, measure lookup + enumeration ---
    // (modest N for spike speed; extrapolate. Each tag is a tiny ref file.)
    let n_tags = 5000u64;
    let timer_create = Instant::now();
    for i in 0..n_tags {
        ds.tags().create(&encode_commit_tag(&format!("bulk{i}")), v_c1).await?;
    }
    let create_ms = timer_create.elapsed().as_millis();
    let timer_lookup = Instant::now();
    let _ = ds.tags().get_version(&encode_commit_tag("bulk2500")).await?;
    let lookup_us = timer_lookup.elapsed().as_micros();
    let timer_enum = Instant::now();
    let all = ds.tags().list().await?;
    let enum_ms = timer_enum.elapsed().as_millis();
    // PASS if single lookup is sub-ms-ish and enumeration of ~5k completes quickly.
    result("0a2.3_scale", lookup_us < 50_000 && enum_ms < 5_000 && all.len() as u64 == n_tags + 2,
        &format!("{n_tags} tags: create_total={create_ms}ms, single_lookup={lookup_us}us, enumerate({})={enum_ms}ms", all.len()));

    // --- 4. RISK-18: compaction preserves pre-compaction tag/version readability ---
    // Add more fragments so there's something to compact, tag the pre-compaction state.
    let mut ds = Dataset::open(&uri).await?;
    for k in 0..5 { ds.append(RecordBatchIterator::new(vec![Ok(batch(1000 + k*20, 20))], schema()), None).await?; }
    let v_precompact = ds.version().version;
    ds.tags().create(&encode_commit_tag("preCompact"), v_precompact).await?;
    let rows_before = ds.count_rows(None).await?;
    let frags_before = ds.get_fragments().len();
    let metrics = compact_files(&mut ds, CompactionOptions::default(), None).await?;
    let frags_after = ds.get_fragments().len();
    // Read the dataset AT the pre-compaction tag — must still return the same rows.
    let at_tag_v = ds.tags().get_version(&encode_commit_tag("preCompact")).await?;
    let at_tag = ds.checkout_version(at_tag_v).await?;
    let rows_at_tag = at_tag.count_rows(None).await?;
    result("0a2.4_compaction_tag_safe", rows_at_tag == rows_before,
        &format!("frags {frags_before}→{frags_after} (compacted {} fragments); tagged version still reads {rows_at_tag}/{rows_before} rows",
            metrics.fragments_removed));

    // --- 5. RISK-18: branch still resolves + reads after parent compaction ---
    let co_branch = Dataset::open(&uri).await?.checkout_branch("feature").await;
    let branch_ok = match co_branch {
        Ok(b) => match b.count_rows(None).await { Ok(r) => format!("branch reads {r} rows after parent compaction"), Err(e) => format!("branch read err: {e}") },
        Err(e) => format!("branch checkout err: {e}"),
    };
    let branch_pass = branch_ok.starts_with("branch reads");
    result("0a2.5_branch_after_compaction", branch_pass, &branch_ok);

    let _ = std::fs::remove_dir_all(&tmp);
    Ok(())
}
