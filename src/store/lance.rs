#![forbid(unsafe_code)]

//! LanceDB-backed persistent store (single-branch linear history).
//!
//! Schema: one row per chunk, keyed by (doc_id, chunk_index).
//! Supports vector search, FTS, and hybrid search. Commit→version binding via
//! Lance tags (managed by layeridx).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance_index::IndexType;
use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
use lance_linalg::distance::DistanceType;
use tokio::sync::RwLock;

use crate::kernel::error::StoreError;
use crate::kernel::model::{
    BranchName, ChunkInfo, Domain, LastIndexed, SearchHit, SearchMode, Statistics, TaskStatus,
};
use crate::layeridx::{self, BranchIndex};

/// A chunk row ready for insertion into Lance.
#[derive(Debug, Clone)]
pub struct ChunkRow {
    pub doc_id: String,
    pub doc_type: String,
    pub chunk_index: i32,
    pub chunk_count: i32,
    pub chunk_token_start: i32,
    pub doc_token_len: i32,
    pub embedding: Vec<f32>,
    pub content: String,
}

/// Search query parameters.
#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub query_embedding: Vec<f32>,
    pub query_text: String,
    pub mode: SearchMode,
    pub start: usize,
    pub count: usize,
    pub doc_type_filter: Vec<String>,
    pub doc_id_filter: Vec<String>,
    pub snippet: bool,
}

/// Whether a ChunkHit's distance is raw (needs transform) or already normalised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceKind {
    /// Raw Lance cosine distance [0, 2] — needs `normalized_cosine_from_lance`.
    RawCosine,
    /// Already normalised to [0, 1] (e.g., from RRF or FTS conversion).
    Normalised,
}

/// Internal chunk-level hit before dedup to documents.
#[derive(Debug, Clone)]
pub struct ChunkHit {
    pub doc_id: String,
    pub distance: f32,
    pub distance_kind: DistanceKind,
    pub chunk_index: i32,
    pub chunk_count: i32,
    pub chunk_token_start: i32,
    pub doc_token_len: i32,
    pub content: String,
}

/// Dataset key: (domain_str, branch_str).
type DatasetKey = (String, String);

/// The Lance-backed store (single-branch linear history).
#[derive(Debug)]
pub struct LanceStore {
    base_dir: PathBuf,
    dim: usize,
    /// Open dataset handles, keyed by (domain, branch).
    datasets: RwLock<HashMap<DatasetKey, Arc<RwLock<Dataset>>>>,
    /// Per-(domain, branch) index tracking.
    branch_indexes: RwLock<HashMap<DatasetKey, BranchIndex>>,
    /// Tasks by task ID.
    tasks: RwLock<HashMap<String, TaskStatus>>,
    /// Per-(domain, branch) pipeline serialisation lock.
    /// Ensures concurrent pushes to the same branch are serialised so that
    /// commit→version tags are correctly isolated.
    pipeline_locks: RwLock<HashMap<DatasetKey, Arc<tokio::sync::Mutex<()>>>>,
}

impl LanceStore {
    /// Create a new LanceStore backed by the given directory.
    pub fn new(base_dir: &Path, dim: usize) -> Self {
        Self {
            base_dir: base_dir.to_owned(),
            dim,
            datasets: RwLock::new(HashMap::new()),
            branch_indexes: RwLock::new(HashMap::new()),
            tasks: RwLock::new(HashMap::new()),
            pipeline_locks: RwLock::new(HashMap::new()),
        }
    }

    /// Acquire the per-(domain, branch) pipeline lock.
    /// Serialises upsert→tag operations so concurrent pushes don't interleave.
    pub async fn acquire_pipeline_lock(
        &self,
        domain: &str,
        branch: &str,
    ) -> tokio::sync::OwnedMutexGuard<()> {
        let key = (domain.to_owned(), branch.to_owned());

        // Get or create the lock for this key.
        let lock = {
            let locks = self.pipeline_locks.read().await;
            if let Some(l) = locks.get(&key) {
                Arc::clone(l)
            } else {
                drop(locks);
                let mut locks = self.pipeline_locks.write().await;
                let l = locks
                    .entry(key)
                    .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                    .clone();
                l
            }
        };

        lock.lock_owned().await
    }

    /// Get the Arrow schema for chunk rows (embedding dimension from config).
    pub fn chunk_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Utf8, false),
            Field::new("doc_type", DataType::Utf8, false),
            Field::new("chunk_index", DataType::Int32, false),
            Field::new("chunk_count", DataType::Int32, false),
            Field::new("chunk_token_start", DataType::Int32, false),
            Field::new("doc_token_len", DataType::Int32, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.dim as i32,
                ),
                false,
            ),
            Field::new("content", DataType::Utf8, false),
        ]))
    }

    /// Get the dataset path for a (domain, branch) pair.
    fn dataset_path(&self, domain: &str, branch: &str) -> PathBuf {
        // Use a safe directory name: domain slashes replaced with double-underscore.
        let safe_domain = domain.replace('/', "__");
        self.base_dir.join(format!("{}_{}.lance", safe_domain, branch))
    }

    /// Open or create the dataset for a (domain, branch) pair.
    pub async fn io_open_dataset(
        &self,
        domain: &str,
        branch: &str,
    ) -> Result<Arc<RwLock<Dataset>>, StoreError> {
        let key = (domain.to_owned(), branch.to_owned());

        // Check cache first.
        {
            let datasets = self.datasets.read().await;
            if let Some(ds) = datasets.get(&key) {
                return Ok(Arc::clone(ds));
            }
        }

        let path = self.dataset_path(domain, branch);
        let uri = path.to_string_lossy().to_string();

        // Try to open existing dataset.
        let ds = if path.exists() {
            Dataset::open(&uri)
                .await
                .map_err(|e| StoreError::Internal(format!("failed to open dataset: {}", e)))?
        } else {
            // Create a new empty dataset with the schema.
            let schema = self.chunk_schema();
            let empty_batch = self.empty_batch();
            let reader = RecordBatchIterator::new(vec![Ok(empty_batch)], schema);
            let params = WriteParams {
                mode: WriteMode::Create,
                ..Default::default()
            };
            Dataset::write(reader, &uri, Some(params))
                .await
                .map_err(|e| StoreError::Internal(format!("failed to create dataset: {}", e)))?
        };

        let arc_ds = Arc::new(RwLock::new(ds));
        let mut datasets = self.datasets.write().await;
        datasets.insert(key, Arc::clone(&arc_ds));
        Ok(arc_ds)
    }

    /// Create an empty RecordBatch with the chunk schema (for dataset initialization).
    fn empty_batch(&self) -> RecordBatch {
        let schema = self.chunk_schema();
        let dim = self.dim as i32;

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(Vec::<&str>::new())),
                Arc::new(StringArray::from(Vec::<&str>::new())),
                Arc::new(Int32Array::from(Vec::<i32>::new())),
                Arc::new(Int32Array::from(Vec::<i32>::new())),
                Arc::new(Int32Array::from(Vec::<i32>::new())),
                Arc::new(Int32Array::from(Vec::<i32>::new())),
                Arc::new(FixedSizeListArray::new(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim,
                    Arc::new(Float32Array::from(Vec::<f32>::new())),
                    None,
                )),
                Arc::new(StringArray::from(Vec::<&str>::new())),
            ],
        )
        .expect("empty batch construction must not fail")
    }

    /// Build a RecordBatch from chunk rows.
    fn rows_to_batch(&self, rows: &[ChunkRow]) -> Result<RecordBatch, StoreError> {
        let schema = self.chunk_schema();

        let doc_ids: Vec<&str> = rows.iter().map(|r| r.doc_id.as_str()).collect();
        let doc_types: Vec<&str> = rows.iter().map(|r| r.doc_type.as_str()).collect();
        let chunk_indexes: Vec<i32> = rows.iter().map(|r| r.chunk_index).collect();
        let chunk_counts: Vec<i32> = rows.iter().map(|r| r.chunk_count).collect();
        let chunk_token_starts: Vec<i32> = rows.iter().map(|r| r.chunk_token_start).collect();
        let doc_token_lens: Vec<i32> = rows.iter().map(|r| r.doc_token_len).collect();
        let contents: Vec<&str> = rows.iter().map(|r| r.content.as_str()).collect();

        // Build the embedding FixedSizeList.
        let flat_embeddings: Vec<f32> = rows.iter().flat_map(|r| r.embedding.iter().copied()).collect();
        let values = Float32Array::from(flat_embeddings);
        let embedding_array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            self.dim as i32,
            Arc::new(values),
            None,
        );

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(doc_ids)),
                Arc::new(StringArray::from(doc_types)),
                Arc::new(Int32Array::from(chunk_indexes)),
                Arc::new(Int32Array::from(chunk_counts)),
                Arc::new(Int32Array::from(chunk_token_starts)),
                Arc::new(Int32Array::from(doc_token_lens)),
                Arc::new(embedding_array) as Arc<dyn arrow_array::Array>,
                Arc::new(StringArray::from(contents)),
            ],
        )
        .map_err(|e| StoreError::Internal(format!("batch construction failed: {}", e)))
    }

    /// Upsert chunk rows for a document. First deletes all existing rows for the
    /// doc_id, then appends the new rows.
    pub async fn io_upsert_chunks(
        &self,
        domain: &str,
        branch: &str,
        doc_id: &str,
        rows: &[ChunkRow],
    ) -> Result<u64, StoreError> {
        if rows.is_empty() {
            return Ok(0);
        }

        let ds_arc = self.io_open_dataset(domain, branch).await?;
        let mut ds = ds_arc.write().await;

        // Delete existing rows for this doc_id.
        let filter = format!("doc_id = '{}'", doc_id.replace('\'', "''"));
        ds.delete(&filter)
            .await
            .map_err(|e| StoreError::Internal(format!("delete failed: {}", e)))?;

        // Append new rows.
        let batch = self.rows_to_batch(rows)?;
        let schema = self.chunk_schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        ds.append(reader, None)
            .await
            .map_err(|e| StoreError::Internal(format!("append failed: {}", e)))?;

        Ok(ds.version().version)
    }

    /// Delete all chunks for a doc_id.
    pub async fn io_delete_doc(
        &self,
        domain: &str,
        branch: &str,
        doc_id: &str,
    ) -> Result<u64, StoreError> {
        let ds_arc = self.io_open_dataset(domain, branch).await?;
        let mut ds = ds_arc.write().await;

        let filter = format!("doc_id = '{}'", doc_id.replace('\'', "''"));
        ds.delete(&filter)
            .await
            .map_err(|e| StoreError::Internal(format!("delete failed: {}", e)))?;

        Ok(ds.version().version)
    }

    /// Bind a commit to a Lance version via a tag.
    pub async fn io_tag_commit(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
        version: u64,
    ) -> Result<(), StoreError> {
        let ds_arc = self.io_open_dataset(domain, branch).await?;
        let ds = ds_arc.read().await;

        let tag = layeridx::encode_commit_tag(commit);
        ds.tags()
            .create(&tag, version)
            .await
            .map_err(|e| StoreError::Internal(format!("tag creation failed: {}", e)))?;

        Ok(())
    }

    /// Ensure an FTS (INVERTED) index exists on the "content" column and is
    /// up-to-date with all fragments. On first call, creates the index; on
    /// subsequent calls, incrementally indexes new (unindexed) fragments via
    /// `optimize_indices` — O(new_data), not O(corpus).
    ///
    /// Lance tracks which fragments are covered by the index via a bitmap.
    /// Queries always scan unindexed fragments via brute-force, so correctness
    /// is guaranteed even before optimize runs. This call improves FTS query
    /// performance by ensuring all fragments are indexed.
    pub async fn io_ensure_fts_index(
        &self,
        domain: &str,
        branch: &str,
    ) -> Result<u64, StoreError> {
        let ds_arc = self.io_open_dataset(domain, branch).await?;
        let mut ds = ds_arc.write().await;

        // Check if the FTS index already exists.
        let indices = ds.load_indices().await
            .map_err(|e| StoreError::Internal(format!("load indices failed: {}", e)))?;
        let has_fts = indices.iter().any(|idx| idx.name == "content_fts");

        if !has_fts {
            // First time: create the inverted index.
            let params = InvertedIndexParams::default();
            ds.create_index(
                &["content"],
                IndexType::Inverted,
                Some("content_fts".to_owned()),
                &params,
                false,
            )
            .await
            .map_err(|e| StoreError::Internal(format!("FTS index creation failed: {}", e)))?;
        } else {
            // Index exists: incrementally index new fragments only.
            ds.optimize_indices(&Default::default())
                .await
                .map_err(|e| StoreError::Internal(format!("FTS index optimize failed: {}", e)))?;
        }

        Ok(ds.version().version)
    }

    /// Resolve a commit to a Lance version via tag lookup.
    pub async fn io_resolve_commit(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
    ) -> Result<Option<u64>, StoreError> {
        let ds_arc = self.io_open_dataset(domain, branch).await?;
        let ds = ds_arc.read().await;

        let tag = layeridx::encode_commit_tag(commit);
        match ds.tags().get_version(&tag).await {
            Ok(v) => Ok(Some(v)),
            Err(_) => Ok(None),
        }
    }

    /// Get last-indexed for a (domain, branch) pair.
    pub async fn last_indexed(&self, domain: &Domain, branch: &BranchName) -> LastIndexed {
        let key = (domain.as_str().to_owned(), branch.as_str().to_owned());
        let indexes = self.branch_indexes.read().await;
        match indexes.get(&key) {
            Some(bi) => LastIndexed {
                branch: branch.as_str().to_owned(),
                commit: bi.commit.clone(),
                version: bi.version,
            },
            None => LastIndexed {
                branch: branch.as_str().to_owned(),
                commit: None,
                version: 0,
            },
        }
    }

    /// Update last-indexed tracking.
    pub async fn update_last_indexed(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
        version: u64,
    ) {
        let key = (domain.to_owned(), branch.to_owned());
        let mut indexes = self.branch_indexes.write().await;
        indexes.insert(
            key,
            BranchIndex {
                commit: Some(commit.to_owned()),
                version,
            },
        );
    }

    /// Record a task status.
    pub async fn record_task(&self, task_id: &str, status: TaskStatus) {
        let mut tasks = self.tasks.write().await;
        tasks.insert(task_id.to_owned(), status);
    }

    /// Check task status.
    pub async fn check_task(&self, task_id: &str) -> Option<TaskStatus> {
        let tasks = self.tasks.read().await;
        tasks.get(task_id).cloned()
    }

    /// Count total rows across all datasets (for statistics).
    pub async fn statistics(&self) -> Statistics {
        let datasets = self.datasets.read().await;
        let indexes = self.branch_indexes.read().await;

        let mut domains = std::collections::HashSet::new();
        let mut branches = 0u64;
        let mut indexed_commits = 0u64;
        let mut chunks = 0u64;
        let mut distinct_docs: std::collections::HashSet<String> = std::collections::HashSet::new();

        for (key, _ds) in datasets.iter() {
            domains.insert(key.0.clone());
            branches += 1;
        }

        for (_key, bi) in indexes.iter() {
            if bi.commit.is_some() {
                indexed_commits += 1;
            }
        }

        // Count rows and distinct doc_ids from datasets (best-effort).
        for (_key, ds_arc) in datasets.iter() {
            let ds = ds_arc.read().await;
            if let Ok(count) = ds.count_rows(None).await {
                chunks += count as u64;
            }
            // Scan doc_id column for distinct count.
            let mut scanner = ds.scan();
            if scanner.project(&["doc_id"]).is_ok() {
                if let Ok(stream) = scanner.try_into_stream().await {
                    if let Ok(batches) = stream.try_collect::<Vec<RecordBatch>>().await {
                        for batch in &batches {
                            if let Some(ids) = batch
                                .column_by_name("doc_id")
                                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                            {
                                for i in 0..ids.len() {
                                    distinct_docs.insert(ids.value(i).to_owned());
                                }
                            }
                        }
                    }
                }
            }
        }

        Statistics {
            domains: domains.len() as u64,
            branches,
            indexed_commits,
            documents: distinct_docs.len() as u64,
            chunks,
        }
    }

    /// Vector/FTS/hybrid search over chunk rows at a given version.
    /// Returns chunk-level hits (caller dedups to documents).
    /// INVARIANT: searches the snapshot at the commit's tagged version, NOT the latest.
    pub async fn io_search(
        &self,
        domain: &str,
        branch: &str,
        commit: &str,
        query: &SearchQuery,
    ) -> Result<Vec<ChunkHit>, StoreError> {
        let ds_arc = self.io_open_dataset(domain, branch).await?;
        let ds = ds_arc.read().await;

        // Resolve commit to version via tag.
        let tag = layeridx::encode_commit_tag(commit);
        let version = ds
            .tags()
            .get_version(&tag)
            .await
            .map_err(|e| StoreError::Internal(format!("commit not indexed: {}", e)))?;

        // Snapshot isolation: checkout the dataset at the resolved version.
        let snapshot = ds
            .checkout_version(version)
            .await
            .map_err(|e| StoreError::Internal(format!("checkout version {} failed: {}", version, e)))?;

        // Build the search based on mode — always against the versioned snapshot.
        let hits = match query.mode {
            SearchMode::Vector => {
                self.vector_search(&snapshot, query).await?
            }
            SearchMode::Fts => {
                self.fts_search(&snapshot, query).await?
            }
            SearchMode::Hybrid => {
                self.hybrid_search(&snapshot, query).await?
            }
        };

        Ok(hits)
    }

    /// Pure vector (ANN) search.
    /// Embeddings are L2-normalised before insert, so L2² distance on unit vectors
    /// equals cosine distance [0, 2]. This gives correct cosine semantics with
    /// Lance's default metric.
    async fn vector_search(
        &self,
        ds: &Dataset,
        query: &SearchQuery,
    ) -> Result<Vec<ChunkHit>, StoreError> {
        // Over-fetch to allow for dedup (multiple chunks per doc).
        let k = (query.start + query.count) * 3;

        let mut scanner = ds.scan();
        scanner
            .nearest("embedding", &Float32Array::from(query.query_embedding.clone()), k)
            .map_err(|e| StoreError::Internal(format!("vector search setup failed: {}", e)))?;
        scanner.distance_metric(DistanceType::Cosine);

        // Apply filters if present.
        if !query.doc_type_filter.is_empty() || !query.doc_id_filter.is_empty() {
            let filter = build_filter_expression(&query.doc_type_filter, &query.doc_id_filter);
            if !filter.is_empty() {
                scanner.filter(&filter)
                    .map_err(|e| StoreError::Internal(format!("filter failed: {}", e)))?;
            }
        }

        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .map_err(|e| StoreError::Internal(format!("scan stream failed: {}", e)))?
            .try_collect()
            .await
            .map_err(|e| StoreError::Internal(format!("batch collect failed: {}", e)))?;

        Ok(batches_to_vector_hits(&batches))
    }

    /// Full-text search.
    /// Returns empty results if no FTS (INVERTED) index exists on the dataset.
    async fn fts_search(
        &self,
        ds: &Dataset,
        query: &SearchQuery,
    ) -> Result<Vec<ChunkHit>, StoreError> {
        let k = (query.start + query.count) * 3;

        let mut scanner = ds.scan();
        // Lance FTS via full_text_search on the "content" column.
        scanner
            .full_text_search(FullTextSearchQuery::new(query.query_text.clone()))
            .map_err(|e| StoreError::Internal(format!("FTS search setup failed: {}", e)))?;
        scanner
            .limit(Some(k as i64), None)
            .map_err(|e| StoreError::Internal(format!("limit failed: {}", e)))?;

        // Apply filters if present.
        if !query.doc_type_filter.is_empty() || !query.doc_id_filter.is_empty() {
            let filter = build_filter_expression(&query.doc_type_filter, &query.doc_id_filter);
            if !filter.is_empty() {
                scanner
                    .filter(&filter)
                    .map_err(|e| StoreError::Internal(format!("filter failed: {}", e)))?;
            }
        }

        let stream = match scanner.try_into_stream().await {
            Ok(s) => s,
            Err(e) => {
                let msg = e.to_string();
                // If no INVERTED index exists, gracefully return empty.
                // This allows hybrid search to degrade to vector-only.
                if msg.contains("INVERTED index") || msg.contains("full text search") {
                    return Ok(Vec::new());
                }
                return Err(StoreError::Internal(format!("FTS stream failed: {}", msg)));
            }
        };

        match stream.try_collect::<Vec<RecordBatch>>().await {
            Ok(batches) => Ok(batches_to_fts_hits(&batches)),
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("INVERTED index") || msg.contains("full text search") {
                    return Ok(Vec::new());
                }
                Err(StoreError::Internal(format!("FTS collect failed: {}", msg)))
            }
        }
    }

    /// Hybrid search: Reciprocal Rank Fusion (RRF) over vector + FTS ranked lists.
    /// Deterministic given deterministic inputs.
    /// RRF score = sum over lists of 1/(k + rank_in_list) where k=60 (standard).
    async fn hybrid_search(
        &self,
        ds: &Dataset,
        query: &SearchQuery,
    ) -> Result<Vec<ChunkHit>, StoreError> {
        let vector_hits = self.vector_search(ds, query).await?;
        let fts_hits = self.fts_search(ds, query).await?;
        Ok(rrf_merge(vector_hits, fts_hits))
    }

    /// Look up a document's chunks by doc_id (indexed lookup for /similar).
    /// INVARIANT: uses a filter on the doc_id column (indexed), NOT a full scan.
    pub async fn io_lookup_doc_chunks(
        &self,
        domain: &str,
        branch: &str,
        doc_id: &str,
    ) -> Result<Vec<ChunkHit>, StoreError> {
        let ds_arc = self.io_open_dataset(domain, branch).await?;
        let ds = ds_arc.read().await;

        let filter = format!("doc_id = '{}'", doc_id.replace('\'', "''"));
        let mut scanner = ds.scan();
        scanner
            .filter(&filter)
            .map_err(|e| StoreError::Internal(format!("lookup filter failed: {}", e)))?;

        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .map_err(|e| StoreError::Internal(format!("lookup stream failed: {}", e)))?
            .try_collect()
            .await
            .map_err(|e| StoreError::Internal(format!("lookup collect failed: {}", e)))?;

        Ok(batches_to_vector_hits(&batches))
    }
}

/// Build a SQL-like filter expression for doc_type and doc_id IN (...) filters.
fn build_filter_expression(doc_types: &[String], doc_ids: &[String]) -> String {
    let mut parts = Vec::new();

    if !doc_types.is_empty() {
        let values: Vec<String> = doc_types
            .iter()
            .map(|t| format!("'{}'", t.replace('\'', "''")))
            .collect();
        parts.push(format!("doc_type IN ({})", values.join(", ")));
    }

    if !doc_ids.is_empty() {
        let values: Vec<String> = doc_ids
            .iter()
            .map(|id| format!("'{}'", id.replace('\'', "''")))
            .collect();
        parts.push(format!("doc_id IN ({})", values.join(", ")));
    }

    // AND across the two sets (OR within each via IN).
    parts.join(" AND ")
}

/// Extract ChunkHit records from RecordBatches (vector search — reads `_distance`).
fn batches_to_vector_hits(batches: &[RecordBatch]) -> Vec<ChunkHit> {
    let mut hits = Vec::new();

    for batch in batches {
        let n = batch.num_rows();
        let doc_ids = batch
            .column_by_name("doc_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let chunk_indexes = batch
            .column_by_name("chunk_index")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let chunk_counts = batch
            .column_by_name("chunk_count")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let chunk_token_starts = batch
            .column_by_name("chunk_token_start")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let doc_token_lens = batch
            .column_by_name("doc_token_len")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let contents = batch
            .column_by_name("content")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let distances = batch
            .column_by_name("_distance")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

        if let (Some(ids), Some(ci), Some(cc), Some(cts), Some(dtl), Some(cnt)) =
            (doc_ids, chunk_indexes, chunk_counts, chunk_token_starts, doc_token_lens, contents)
        {
            for i in 0..n {
                let distance = distances.map(|d| d.value(i)).unwrap_or(0.0);
                hits.push(ChunkHit {
                    doc_id: ids.value(i).to_owned(),
                    distance,
                    distance_kind: DistanceKind::RawCosine,
                    chunk_index: ci.value(i),
                    chunk_count: cc.value(i),
                    chunk_token_start: cts.value(i),
                    doc_token_len: dtl.value(i),
                    content: cnt.value(i).to_owned(),
                });
            }
        }
    }

    hits
}

/// Extract ChunkHit records from RecordBatches (FTS search — reads `_score`).
/// BM25 scores are converted to distances: `distance = 1/(1+score)` so 0=best.
/// Results are preserved in stream order (BM25 rank order from Lance).
fn batches_to_fts_hits(batches: &[RecordBatch]) -> Vec<ChunkHit> {
    let mut hits = Vec::new();

    for batch in batches {
        let n = batch.num_rows();
        let doc_ids = batch
            .column_by_name("doc_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let chunk_indexes = batch
            .column_by_name("chunk_index")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let chunk_counts = batch
            .column_by_name("chunk_count")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let chunk_token_starts = batch
            .column_by_name("chunk_token_start")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let doc_token_lens = batch
            .column_by_name("doc_token_len")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let contents = batch
            .column_by_name("content")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let scores = batch
            .column_by_name("_score")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

        if let (Some(ids), Some(ci), Some(cc), Some(cts), Some(dtl), Some(cnt)) =
            (doc_ids, chunk_indexes, chunk_counts, chunk_token_starts, doc_token_lens, contents)
        {
            for i in 0..n {
                // BM25 score → distance: higher score = more relevant = lower distance.
                let score = scores.map(|s| s.value(i)).unwrap_or(0.0);
                let distance = 1.0 / (1.0 + score);
                hits.push(ChunkHit {
                    doc_id: ids.value(i).to_owned(),
                    distance,
                    distance_kind: DistanceKind::Normalised,
                    chunk_index: ci.value(i),
                    chunk_count: cc.value(i),
                    chunk_token_start: cts.value(i),
                    doc_token_len: dtl.value(i),
                    content: cnt.value(i).to_owned(),
                });
            }
        }
    }

    hits
}

/// Reciprocal Rank Fusion: merge two ranked lists into a single list.
/// RRF score = sum of 1/(k + rank) across all lists where the item appears.
/// k=60 is the standard constant. Items are keyed by (doc_id, chunk_index).
/// The output is sorted by descending RRF score (highest relevance first).
/// Distance is set to 1.0 - normalized_rrf_score (so smaller = more relevant).
fn rrf_merge(vector_hits: Vec<ChunkHit>, fts_hits: Vec<ChunkHit>) -> Vec<ChunkHit> {
    use std::collections::HashMap;
    const K: f32 = 60.0;

    // Key: (doc_id, chunk_index) → (rrf_score, best ChunkHit)
    let mut scores: HashMap<(String, i32), (f32, ChunkHit)> = HashMap::new();

    // Score vector results by rank (already sorted by distance — best first).
    for (rank, hit) in vector_hits.into_iter().enumerate() {
        let key = (hit.doc_id.clone(), hit.chunk_index);
        let rrf_score = 1.0 / (K + rank as f32 + 1.0);
        let entry = scores.entry(key).or_insert_with(|| (0.0, hit.clone()));
        entry.0 += rrf_score;
        // Keep the hit with the better (smaller) original distance.
        if hit.distance < entry.1.distance {
            entry.1 = hit;
        }
    }

    // Score FTS results by rank (already sorted by relevance — best first).
    for (rank, hit) in fts_hits.into_iter().enumerate() {
        let key = (hit.doc_id.clone(), hit.chunk_index);
        let rrf_score = 1.0 / (K + rank as f32 + 1.0);
        let entry = scores.entry(key).or_insert_with(|| (0.0, hit.clone()));
        entry.0 += rrf_score;
        if hit.distance < entry.1.distance {
            entry.1 = hit;
        }
    }

    // Sort by RRF score descending (highest relevance first).
    let mut ranked: Vec<(f32, ChunkHit)> = scores.into_values().collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Convert RRF score into a synthetic distance (lower = better).
    // Normalise: max possible single-list score is 1/(K+1) ≈ 0.0164.
    // Max combined (rank 1 in both) is 2/(K+1) ≈ 0.0328.
    // We map to [0, 1] via: distance = 1.0 - (score / max_score).
    let max_score = 2.0 / (K + 1.0);
    ranked
        .into_iter()
        .map(|(score, mut hit)| {
            hit.distance = (1.0 - score / max_score).clamp(0.0, 1.0);
            hit.distance_kind = DistanceKind::Normalised;
            hit
        })
        .collect()
}

/// Dedup chunk-level hits to document-level hits (best chunk per doc_id).
/// Distance transform is mode-aware: only RawCosine hits get `normalized_cosine_from_lance`;
/// already-normalised hits (FTS, RRF) pass through unchanged.
pub fn dedup_chunks_to_documents(hits: Vec<ChunkHit>, snippet: bool) -> Vec<SearchHit> {
    use std::collections::HashMap;
    use crate::kernel::distance::normalized_cosine_from_lance;
    use crate::chunk::chunk_location;

    // Group by doc_id, keep the best (smallest distance) chunk per document.
    let mut best_per_doc: HashMap<String, ChunkHit> = HashMap::new();
    for hit in hits {
        let entry = best_per_doc
            .entry(hit.doc_id.clone())
            .or_insert_with(|| hit.clone());
        if hit.distance < entry.distance {
            *entry = hit;
        }
    }

    let mut results: Vec<SearchHit> = best_per_doc
        .into_values()
        .map(|hit| {
            let location = chunk_location(hit.chunk_token_start as u32, hit.doc_token_len as u32);
            let final_distance = match hit.distance_kind {
                DistanceKind::RawCosine => normalized_cosine_from_lance(hit.distance),
                DistanceKind::Normalised => hit.distance,
            };
            SearchHit {
                id: hit.doc_id,
                distance: final_distance,
                chunk: ChunkInfo {
                    index: hit.chunk_index as u32,
                    count: hit.chunk_count as u32,
                    token_start: hit.chunk_token_start as u32,
                    doc_token_len: hit.doc_token_len as u32,
                    location,
                    snippet: if snippet { Some(hit.content) } else { None },
                },
            }
        })
        .collect();

    // Sort by distance (nearest first).
    results.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

#[cfg(test)]
#[allow(clippy::useless_vec)]
mod tests {
    use super::*;

    fn make_test_store(dim: usize) -> (LanceStore, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let store = LanceStore::new(tmp.path(), dim);
        (store, tmp)
    }

    fn fake_embedding(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| (seed + i as f32 * 0.01).sin()).collect()
    }

    // --- upsert chunks, tag commit, verify round-trip ---
    #[tokio::test]
    async fn upsert_and_tag_commit_round_trips() {
        let (store, _tmp) = make_test_store(8);
        let rows = vec![
            ChunkRow {
                doc_id: "doc/1".to_owned(),
                doc_type: "People".to_owned(),
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 40,
                embedding: fake_embedding(8, 1.0),
                content: "Yoda is a wise Jedi.".to_owned(),
            },
            ChunkRow {
                doc_id: "doc/2".to_owned(),
                doc_type: "Species".to_owned(),
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 30,
                embedding: fake_embedding(8, 2.0),
                content: "Mon Calamari are squid people.".to_owned(),
            },
        ];

        let version = store
            .io_upsert_chunks("admin/star_wars", "main", "doc/1", &rows[0..1])
            .await
            .expect("upsert doc/1");
        assert!(version > 0);

        let version2 = store
            .io_upsert_chunks("admin/star_wars", "main", "doc/2", &rows[1..2])
            .await
            .expect("upsert doc/2");
        assert!(version2 > version);

        // Tag commit c0 to the final version.
        store
            .io_tag_commit("admin/star_wars", "main", "c0", version2)
            .await
            .expect("tag commit");

        // Resolve commit should return the version.
        let resolved = store
            .io_resolve_commit("admin/star_wars", "main", "c0")
            .await
            .expect("resolve");
        assert_eq!(resolved, Some(version2));
    }

    // --- multi-chunk doc produces multiple rows ---
    #[tokio::test]
    async fn multi_chunk_doc_produces_multiple_rows() {
        let (store, _tmp) = make_test_store(8);
        let rows = vec![
            ChunkRow {
                doc_id: "doc/big".to_owned(),
                doc_type: "Article".to_owned(),
                chunk_index: 0,
                chunk_count: 3,
                chunk_token_start: 0,
                doc_token_len: 1500,
                embedding: fake_embedding(8, 1.0),
                content: "Beginning of the article.".to_owned(),
            },
            ChunkRow {
                doc_id: "doc/big".to_owned(),
                doc_type: "Article".to_owned(),
                chunk_index: 1,
                chunk_count: 3,
                chunk_token_start: 450,
                doc_token_len: 1500,
                embedding: fake_embedding(8, 2.0),
                content: "Middle of the article.".to_owned(),
            },
            ChunkRow {
                doc_id: "doc/big".to_owned(),
                doc_type: "Article".to_owned(),
                chunk_index: 2,
                chunk_count: 3,
                chunk_token_start: 900,
                doc_token_len: 1500,
                embedding: fake_embedding(8, 3.0),
                content: "End of the article.".to_owned(),
            },
        ];

        store
            .io_upsert_chunks("admin/test", "main", "doc/big", &rows)
            .await
            .expect("upsert multi-chunk");

        // Lookup should find all 3 chunks.
        let chunks = store
            .io_lookup_doc_chunks("admin/test", "main", "doc/big")
            .await
            .expect("lookup");
        assert_eq!(chunks.len(), 3);
    }

    // --- delete removes all chunks ---
    #[tokio::test]
    async fn delete_doc_removes_all_chunks() {
        let (store, _tmp) = make_test_store(8);
        let rows = vec![
            ChunkRow {
                doc_id: "doc/del".to_owned(),
                doc_type: "X".to_owned(),
                chunk_index: 0,
                chunk_count: 2,
                chunk_token_start: 0,
                doc_token_len: 100,
                embedding: fake_embedding(8, 1.0),
                content: "part 1".to_owned(),
            },
            ChunkRow {
                doc_id: "doc/del".to_owned(),
                doc_type: "X".to_owned(),
                chunk_index: 1,
                chunk_count: 2,
                chunk_token_start: 50,
                doc_token_len: 100,
                embedding: fake_embedding(8, 2.0),
                content: "part 2".to_owned(),
            },
        ];

        store
            .io_upsert_chunks("admin/test", "main", "doc/del", &rows)
            .await
            .expect("upsert");

        store
            .io_delete_doc("admin/test", "main", "doc/del")
            .await
            .expect("delete");

        let remaining = store
            .io_lookup_doc_chunks("admin/test", "main", "doc/del")
            .await
            .expect("lookup after delete");
        assert_eq!(remaining.len(), 0, "all chunks should be deleted");
    }

    // --- dedup produces correct chunk metadata ---
    #[test]
    fn dedup_chunks_to_documents_picks_best_chunk() {
        let hits = vec![
            ChunkHit {
                doc_id: "doc/1".to_owned(),
                distance: 0.8,
                distance_kind: DistanceKind::RawCosine,
                chunk_index: 0,
                chunk_count: 2,
                chunk_token_start: 0,
                doc_token_len: 1000,
                content: "first chunk".to_owned(),
            },
            ChunkHit {
                doc_id: "doc/1".to_owned(),
                distance: 0.4, // Better (smaller distance) — this chunk wins.
                distance_kind: DistanceKind::RawCosine,
                chunk_index: 1,
                chunk_count: 2,
                chunk_token_start: 500,
                doc_token_len: 1000,
                content: "second chunk".to_owned(),
            },
            ChunkHit {
                doc_id: "doc/2".to_owned(),
                distance: 0.6,
                distance_kind: DistanceKind::RawCosine,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 200,
                content: "only chunk".to_owned(),
            },
        ];

        let results = dedup_chunks_to_documents(hits, false);
        assert_eq!(results.len(), 2, "should dedup to 2 documents");

        // Results sorted by distance — doc/1 (0.4→0.2 after transform) < doc/2 (0.6→0.3).
        let doc1 = results.iter().find(|r| r.id == "doc/1").expect("doc/1");
        assert_eq!(doc1.chunk.index, 1, "best chunk is index 1");
        assert_eq!(doc1.chunk.count, 2);
        assert_eq!(doc1.chunk.token_start, 500);
        assert_eq!(doc1.chunk.doc_token_len, 1000);
        // location = 500/1000 = 0.5
        assert!((doc1.chunk.location - 0.5).abs() < f32::EPSILON);
        assert!(doc1.chunk.snippet.is_none(), "snippet should be omitted");
    }

    // --- single chunk doc has location 0.0 ---
    #[test]
    fn dedup_single_chunk_doc_location_zero() {
        let hits = vec![ChunkHit {
            doc_id: "doc/s".to_owned(),
            distance: 0.2,
            distance_kind: DistanceKind::RawCosine,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 41,
            content: "short doc".to_owned(),
        }];

        let results = dedup_chunks_to_documents(hits, true);
        assert_eq!(results.len(), 1);
        let hit = &results[0];
        assert_eq!(hit.chunk.index, 0);
        assert_eq!(hit.chunk.count, 1);
        assert_eq!(hit.chunk.token_start, 0);
        assert_eq!(hit.chunk.doc_token_len, 41);
        assert_eq!(hit.chunk.location, 0.0);
        assert_eq!(hit.chunk.snippet, Some("short doc".to_owned()));
    }

    // --- distance transform applied correctly ---
    #[test]
    fn distance_transform_in_dedup() {
        let hits = vec![ChunkHit {
            doc_id: "doc/x".to_owned(),
            distance: 0.0, // Self-distance in lance cosine.
            distance_kind: DistanceKind::RawCosine,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "x".to_owned(),
        }];
        let results = dedup_chunks_to_documents(hits, false);
        assert_eq!(results[0].distance, 0.0, "self-distance maps to 0");
    }

    // --- #3: normalised distances skip the transform in dedup ---
    #[test]
    fn dedup_normalised_distances_pass_through() {
        let hits = vec![
            ChunkHit {
                doc_id: "doc/rrf".to_owned(),
                distance: 0.42, // Already normalised (e.g., from RRF).
                distance_kind: DistanceKind::Normalised,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 10,
                content: "rrf hit".to_owned(),
            },
        ];
        let results = dedup_chunks_to_documents(hits, false);
        // Must pass through unchanged — NOT halved by normalized_cosine_from_lance.
        assert!(
            (results[0].distance - 0.42).abs() < f32::EPSILON,
            "normalised distance should pass through unchanged, got {}",
            results[0].distance,
        );
    }

    // --- #1: FTS distances are non-zero and ordered (BM25 score preserved) ---
    #[test]
    fn fts_hits_have_nonzero_ordered_distances() {
        // Simulate FTS hits with BM25 scores converted to distances.
        let hits = vec![
            ChunkHit {
                doc_id: "doc/best".to_owned(),
                distance: 1.0 / (1.0 + 10.0), // High BM25 score → low distance.
                distance_kind: DistanceKind::Normalised,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 10,
                content: "best match".to_owned(),
            },
            ChunkHit {
                doc_id: "doc/worse".to_owned(),
                distance: 1.0 / (1.0 + 2.0), // Lower BM25 score → higher distance.
                distance_kind: DistanceKind::Normalised,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 10,
                content: "worse match".to_owned(),
            },
        ];

        let results = dedup_chunks_to_documents(hits, false);
        assert_eq!(results.len(), 2);
        // Best match (lowest distance) should be first after sorting.
        assert_eq!(results[0].id, "doc/best");
        assert_eq!(results[1].id, "doc/worse");
        // Both distances must be non-zero.
        assert!(results[0].distance > 0.0, "FTS distance must be > 0");
        assert!(results[1].distance > 0.0, "FTS distance must be > 0");
        // Best has lower distance.
        assert!(results[0].distance < results[1].distance);
    }

    // --- #2: Vector distance scale anchors (locks factor-of-2 correctness) ---
    // With DistanceType::Cosine on the scanner, _distance is a true cosine distance
    // in [0,2]. normalized_cosine_from_lance(d) = d/2 maps to [0,1].
    // These anchors catch any factor-of-2 scale bug permanently.
    #[test]
    fn vector_distance_scale_anchors_through_dedup() {
        // Anchor 1: self-distance (identical vectors) → 0.0
        let hit_identical = ChunkHit {
            doc_id: "doc/self".to_owned(),
            distance: 0.0, // Lance cosine: identical vectors
            distance_kind: DistanceKind::RawCosine,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "self".to_owned(),
        };
        let results = dedup_chunks_to_documents(vec![hit_identical], false);
        assert_eq!(results[0].distance, 0.0, "identical → 0.0");

        // Anchor 2: orthogonal vectors (Lance cosine distance = 1.0) → 0.5
        let hit_orthogonal = ChunkHit {
            doc_id: "doc/ortho".to_owned(),
            distance: 1.0, // Lance cosine: orthogonal
            distance_kind: DistanceKind::RawCosine,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "ortho".to_owned(),
        };
        let results = dedup_chunks_to_documents(vec![hit_orthogonal], false);
        assert!(
            (results[0].distance - 0.5).abs() < f32::EPSILON,
            "orthogonal → 0.5, got {}",
            results[0].distance,
        );

        // Anchor 3: opposite vectors (Lance cosine distance = 2.0) → 1.0
        let hit_opposite = ChunkHit {
            doc_id: "doc/opp".to_owned(),
            distance: 2.0, // Lance cosine: opposite
            distance_kind: DistanceKind::RawCosine,
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            content: "opposite".to_owned(),
        };
        let results = dedup_chunks_to_documents(vec![hit_opposite], false);
        assert_eq!(results[0].distance, 1.0, "opposite → 1.0");

        // The OLD bug: L2² for orthogonal unit vectors = 2.0, which would give
        // normalized_cosine_from_lance(2.0) = 1.0 (WRONG — should be 0.5).
        // This test catches any regression where L2² is fed to the transform.
    }

    // --- statistics reflect indexed data ---
    #[tokio::test]
    async fn statistics_reflect_indexed_data() {
        let (store, _tmp) = make_test_store(8);
        let rows = vec![ChunkRow {
            doc_id: "doc/1".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: fake_embedding(8, 1.0),
            content: "test".to_owned(),
        }];

        store
            .io_upsert_chunks("admin/db", "main", "doc/1", &rows)
            .await
            .expect("upsert");
        store.update_last_indexed("admin/db", "main", "c0", 1).await;

        let stats = store.statistics().await;
        assert!(stats.chunks > 0, "chunks should be > 0 after upsert");
        assert!(stats.domains > 0, "domains should be > 0");
        assert!(stats.indexed_commits > 0, "indexed_commits should be > 0");
    }

    // --- commit tag isolation (different tags for different versions) ---
    #[tokio::test]
    async fn different_commits_different_versions() {
        let (store, _tmp) = make_test_store(8);

        // Insert first doc, tag as c0.
        let rows1 = vec![ChunkRow {
            doc_id: "doc/1".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: fake_embedding(8, 1.0),
            content: "version one".to_owned(),
        }];
        let v1 = store
            .io_upsert_chunks("admin/db", "main", "doc/1", &rows1)
            .await
            .expect("upsert v1");
        store
            .io_tag_commit("admin/db", "main", "c0", v1)
            .await
            .expect("tag c0");

        // Insert second doc, tag as c1.
        let rows2 = vec![ChunkRow {
            doc_id: "doc/2".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: fake_embedding(8, 2.0),
            content: "version two".to_owned(),
        }];
        let v2 = store
            .io_upsert_chunks("admin/db", "main", "doc/2", &rows2)
            .await
            .expect("upsert v2");
        store
            .io_tag_commit("admin/db", "main", "c1", v2)
            .await
            .expect("tag c1");

        // Resolve both — different versions.
        let r0 = store.io_resolve_commit("admin/db", "main", "c0").await.unwrap();
        let r1 = store.io_resolve_commit("admin/db", "main", "c1").await.unwrap();
        assert_eq!(r0, Some(v1));
        assert_eq!(r1, Some(v2));
        assert_ne!(v1, v2, "versions must differ");
    }

    // --- snapshot isolation: search at C0 does not see C1 data ---
    #[tokio::test]
    async fn snapshot_isolation_search_at_old_commit_excludes_new_data() {
        let (store, _tmp) = make_test_store(8);

        // Insert doc/A, tag as commit "c0".
        let emb_a = fake_embedding(8, 1.0);
        let rows_a = vec![ChunkRow {
            doc_id: "doc/A".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: emb_a.clone(),
            content: "document A".to_owned(),
        }];
        let v0 = store
            .io_upsert_chunks("admin/iso", "main", "doc/A", &rows_a)
            .await
            .expect("upsert A");
        store
            .io_tag_commit("admin/iso", "main", "c0", v0)
            .await
            .expect("tag c0");

        // Insert doc/B, tag as commit "c1".
        let emb_b = fake_embedding(8, 2.0);
        let rows_b = vec![ChunkRow {
            doc_id: "doc/B".to_owned(),
            doc_type: "T".to_owned(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_token_start: 0,
            doc_token_len: 10,
            embedding: emb_b.clone(),
            content: "document B".to_owned(),
        }];
        let v1 = store
            .io_upsert_chunks("admin/iso", "main", "doc/B", &rows_b)
            .await
            .expect("upsert B");
        store
            .io_tag_commit("admin/iso", "main", "c1", v1)
            .await
            .expect("tag c1");

        // Search at c0 — should only find doc/A.
        let query_c0 = SearchQuery {
            query_embedding: emb_a.clone(),
            query_text: "document".to_owned(),
            mode: crate::kernel::model::SearchMode::Vector,
            start: 0,
            count: 10,
            doc_type_filter: Vec::new(),
            doc_id_filter: Vec::new(),
            snippet: false,
        };
        let hits_c0 = store
            .io_search("admin/iso", "main", "c0", &query_c0)
            .await
            .expect("search at c0");

        let doc_ids_c0: Vec<&str> = hits_c0.iter().map(|h| h.doc_id.as_str()).collect();
        assert!(
            doc_ids_c0.contains(&"doc/A"),
            "c0 snapshot should contain doc/A"
        );
        assert!(
            !doc_ids_c0.contains(&"doc/B"),
            "c0 snapshot must NOT contain doc/B (added after c0)"
        );

        // Search at c1 — should find both doc/A and doc/B.
        let query_c1 = SearchQuery {
            query_embedding: emb_a,
            query_text: "document".to_owned(),
            mode: crate::kernel::model::SearchMode::Vector,
            start: 0,
            count: 10,
            doc_type_filter: Vec::new(),
            doc_id_filter: Vec::new(),
            snippet: false,
        };
        let hits_c1 = store
            .io_search("admin/iso", "main", "c1", &query_c1)
            .await
            .expect("search at c1");

        let doc_ids_c1: Vec<&str> = hits_c1.iter().map(|h| h.doc_id.as_str()).collect();
        assert!(
            doc_ids_c1.contains(&"doc/A"),
            "c1 snapshot should contain doc/A"
        );
        assert!(
            doc_ids_c1.contains(&"doc/B"),
            "c1 snapshot should contain doc/B"
        );
    }

    // --- RRF merge produces correct ranking ---
    #[test]
    fn rrf_merge_combines_ranked_lists() {
        // Vector ranked: A (best), B, C
        let vector_hits = vec![
            ChunkHit {
                doc_id: "A".to_owned(),
                distance: 0.1,
                distance_kind: DistanceKind::RawCosine,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 10,
                content: "a".to_owned(),
            },
            ChunkHit {
                doc_id: "B".to_owned(),
                distance: 0.3,
                distance_kind: DistanceKind::RawCosine,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 10,
                content: "b".to_owned(),
            },
            ChunkHit {
                doc_id: "C".to_owned(),
                distance: 0.5,
                distance_kind: DistanceKind::RawCosine,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 10,
                content: "c".to_owned(),
            },
        ];

        // FTS ranked: B (best), C, D (new — only in FTS)
        let fts_hits = vec![
            ChunkHit {
                doc_id: "B".to_owned(),
                distance: 0.1, // FTS distance (from BM25 conversion).
                distance_kind: DistanceKind::Normalised,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 10,
                content: "b".to_owned(),
            },
            ChunkHit {
                doc_id: "C".to_owned(),
                distance: 0.2,
                distance_kind: DistanceKind::Normalised,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 10,
                content: "c".to_owned(),
            },
            ChunkHit {
                doc_id: "D".to_owned(),
                distance: 0.3,
                distance_kind: DistanceKind::Normalised,
                chunk_index: 0,
                chunk_count: 1,
                chunk_token_start: 0,
                doc_token_len: 10,
                content: "d".to_owned(),
            },
        ];

        let merged = rrf_merge(vector_hits, fts_hits);

        // B should be ranked highest: rank 2 in vector + rank 1 in FTS
        // = 1/(60+2) + 1/(60+1) = 1/62 + 1/61
        assert_eq!(merged[0].doc_id, "B", "B should rank first (appears high in both lists)");

        // All 4 unique docs should appear.
        let ids: Vec<&str> = merged.iter().map(|h| h.doc_id.as_str()).collect();
        assert!(ids.contains(&"A"));
        assert!(ids.contains(&"B"));
        assert!(ids.contains(&"C"));
        assert!(ids.contains(&"D"));
        assert_eq!(ids.len(), 4);
    }
}
