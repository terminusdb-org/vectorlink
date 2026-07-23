// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

//! Write pipeline: upsert, batch delete+append, delete, compaction.

use std::sync::Arc;

use arrow_array::RecordBatchIterator;
use lance::dataset::Dataset;
use lance::dataset::write::DeleteBuilder;
use lance::deps::datafusion::logical_expr::{col, lit, in_list};

use crate::kernel::error::StoreError;

use super::{ChunkRow, LanceStore};

impl LanceStore {
    /// Upsert chunk rows for a document on `branch` (layout A). First deletes all
    /// existing rows for the doc_id, then appends the new rows — the
    /// delete-then-append that implements real `Changed` (replace full chunk set,
    /// no stale chunks; RISK-13).
    ///
    /// Writes target `branch`'s head via a branch-bound handle so sibling
    /// branches are untouched. Ensures the domain dataset exists first.
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

        // Serialise dataset creation against DELETE /domain (BLOCKER-2 / #6):
        // hold the per-domain guard across the ensure-exists so a concurrent
        // delete can't observe a half-created dataset.
        let _domain_guard = self.acquire_domain_guard(domain).await;

        // Ensure the domain dataset exists (creates the main branch on first use).
        self.io_open_dataset(domain, branch).await?;

        // Open a fresh branch-bound handle so the write targets `branch`'s head.
        let mut ds = self.io_open_branch_for_write(domain, branch).await?;

        // Delete existing rows for this doc_id (replace semantics).
        let expr = col("doc_id").eq(lit(doc_id));
        let result = DeleteBuilder::from_expr(Arc::new(ds.clone()), expr)
            .execute()
            .await
            .map_err(|e| StoreError::Internal(format!("delete failed: {}", e)))?;
        ds = result.new_dataset.as_ref().clone();

        // Append new rows.
        let batch = self.rows_to_batch(rows)?;
        let schema = self.chunk_schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        ds.append(reader, None)
            .await
            .map_err(|e| StoreError::Internal(format!("append failed: {}", e)))?;

        // Keep the cache consistent with the branch we just advanced (main).
        // For non-main branches the cached main handle is unaffected; refreshing
        // is harmless (re-opens at default branch head).
        self.io_refresh_cached_dataset(domain, branch).await?;

        Ok(ds.version().version)
    }

    /// Batched write for one commit's operations (Phase 6A Step 1).
    ///
    /// Performs at most TWO Lance version advances in a fixed, crash-safe order:
    ///   1. (Optional) Delete: `ds.delete(doc_id IN (...))` for all docs that are
    ///      being replaced (Changed) or removed (Deleted). Creates deletion vectors
    ///      only — no new data fragment. Handles shrinking chunk counts correctly
    ///      (all old chunks for a doc_id are removed regardless of count change).
    ///   2. (Optional) Append: `ds.append(batch)` with ALL new rows for Insert +
    ///      Changed docs. Exactly ONE new data fragment for all rows.
    ///
    /// Crash safety (two-version delete-then-append, Option B):
    ///   This is NOT an atomic single-version merge-insert. It performs two separate
    ///   Lance version advances (delete, then append). Crash-safety comes from the
    ///   untagged-commit→invisible→re-pushable property:
    ///   - Crash before (1): no change. Commit remains untagged → invisible → re-pushable.
    ///   - Crash after (1) but before (2): Changed/Deleted docs' old chunks removed
    ///     (correct for Deleted; for Changed, commit untagged → invisible → re-pushable,
    ///     the re-push will re-insert the replacement rows).
    ///   - Crash after (2): both writes committed. Tag not yet written → re-push is
    ///     idempotent (delete of already-deleted rows is no-op; append of same rows
    ///     would duplicate, but since commit is untagged it was never served →
    ///     re-push via the pipeline lock clears and retries correctly).
    ///
    /// Caller MUST hold the pipeline lock. Dataset existence is ensured internally.
    ///
    /// Returns the final Lance version after all writes.
    pub async fn io_batch_delete_append(
        &self,
        domain: &str,
        branch: &str,
        delete_ids: &[String],
        rows: &[ChunkRow],
    ) -> Result<u64, StoreError> {
        // Serialise dataset creation against DELETE /domain (BLOCKER-2 / #6):
        let _domain_guard = self.acquire_domain_guard(domain).await;

        // Ensure the domain dataset exists (creates the main branch on first use).
        self.io_open_dataset(domain, branch).await?;

        // Use a SINGLE mutable handle for both delete and append. This ensures
        // the append version is built ON TOP of the post-delete version (the append
        // inherits the deletion vectors). Using separate handles would risk the
        // second Dataset::open returning a stale version if the filesystem hasn't
        // flushed the latest_version_hint before the second open reads it.
        let mut ds = self.io_open_branch_for_write(domain, branch).await?;

        // --- Phase 1: delete old chunks for Changed + Deleted doc_ids ---
        // Creates deletion vectors only (no new data fragment). Handles shrinking
        // chunk counts: ALL rows for a doc_id are removed regardless of old/new count.
        if !delete_ids.is_empty() {
            let values: Vec<_> = delete_ids.iter().map(|id| lit(id.as_str())).collect();
            let expr = in_list(col("doc_id"), values, false);
            let result = DeleteBuilder::from_expr(Arc::new(ds.clone()), expr)
                .execute()
                .await
                .map_err(|e| StoreError::Internal(format!("batch delete failed: {}", e)))?;
            ds = result.new_dataset.as_ref().clone();
        }

        // --- Phase 2: append all new rows (one fragment for all Insert + Changed) ---
        // Since Phase 1 already deleted old rows for Changed docs, this is a pure
        // append — no merge semantics needed. Produces exactly ONE new data fragment.
        // The append builds on the post-delete version (same handle), so the final
        // manifest correctly includes the deletion vectors.
        if !rows.is_empty() {
            let batch = self.rows_to_batch(rows)?;
            let schema = self.chunk_schema();
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
            ds.append(reader, None)
                .await
                .map_err(|e| StoreError::Internal(format!("batch append failed: {}", e)))?;
        }

        // Capture the version from the write handle BEFORE refreshing the cache.
        // This is the authoritative post-write version (delete + append).
        let version = ds.version().version;

        // Refresh the cached handle so subsequent reads see the new data.
        self.io_refresh_cached_dataset(domain, branch).await?;

        Ok(version)
    }

    /// Append a micro-batch of rows to an already-open write handle.
    /// Used by the streaming pipeline to write incrementally without
    /// accumulating all rows in memory. Each call produces one Lance fragment.
    /// The caller MUST hold the pipeline lock and the domain guard.
    pub async fn io_microbatch_append(
        &self,
        ds: &mut Dataset,
        rows: &[ChunkRow],
    ) -> Result<(), StoreError> {
        if rows.is_empty() {
            return Ok(());
        }
        let batch = self.rows_to_batch(rows)?;
        let schema = self.chunk_schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        ds.append(reader, None)
            .await
            .map_err(|e| StoreError::Internal(format!("microbatch append failed: {}", e)))?;
        Ok(())
    }

    /// Delete old chunks for a set of doc_ids on an already-open write handle.
    /// Used by the streaming pipeline to do the delete phase once at the start
    /// before micro-batch appends begin.
    pub async fn io_microbatch_delete(
        &self,
        ds: &mut Dataset,
        delete_ids: &[String],
    ) -> Result<(), StoreError> {
        if delete_ids.is_empty() {
            return Ok(());
        }
        let values: Vec<_> = delete_ids.iter().map(|id| lit(id.as_str())).collect();
        let expr = in_list(col("doc_id"), values, false);
        let result = DeleteBuilder::from_expr(Arc::new(ds.clone()), expr)
            .execute()
            .await
            .map_err(|e| StoreError::Internal(format!("microbatch delete failed: {}", e)))?;
        *ds = result.new_dataset.as_ref().clone();
        Ok(())
    }

    /// Delete all chunks for a doc_id on `branch` (`Deleted` op; RISK-13).
    /// Writes target `branch`'s head via a branch-bound handle.
    pub async fn io_delete_doc(
        &self,
        domain: &str,
        branch: &str,
        doc_id: &str,
    ) -> Result<u64, StoreError> {
        // A delete against a domain that does not exist is a no-op — it must NOT
        // create the dataset (BLOCKER-2 resurrection guard). Only a genuine
        // insert/change creates a domain.
        if self.io_open_dataset_readonly(domain).await?.is_none() {
            return Ok(0);
        }

        let mut ds = self.io_open_branch_for_write(domain, branch).await?;

        let expr = col("doc_id").eq(lit(doc_id));
        let result = DeleteBuilder::from_expr(Arc::new(ds.clone()), expr)
            .execute()
            .await
            .map_err(|e| StoreError::Internal(format!("delete failed: {}", e)))?;
        ds = result.new_dataset.as_ref().clone();

        self.io_refresh_cached_dataset(domain, branch).await?;

        Ok(ds.version().version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the same Expr used by io_upsert_chunks and io_delete_doc
    /// for individual doc_id deletion — demonstrates injection-safe construction.
    fn build_single_expr(doc_id: &str) -> lance::deps::datafusion::logical_expr::Expr {
        col("doc_id").eq(lit(doc_id))
    }

    #[test]
    fn filter_normal_iri_is_safe() {
        // Normal IRIs produce a simple equality Expr — no SQL string involved.
        let expr = build_single_expr("terminusdb:///db/People/123");
        let rendered = format!("{}", expr);
        assert!(rendered.contains("terminusdb:///db/People/123"));
    }

    #[test]
    fn filter_backslash_quote_is_safe_via_expr() {
        // Previously, a doc_id like x\' OR 1=1 could break out of a SQL string
        // literal via backslash escaping. With Expr-based construction, the
        // entire doc_id is a literal value — no SQL parsing occurs.
        let malicious = "x\\' OR 1=1";
        let expr = build_single_expr(malicious);
        let rendered = format!("{}", expr);
        // The malicious content is contained within the literal — no SQL injection.
        assert!(rendered.contains("OR 1=1"));
        // It's part of the literal value, not injected SQL.
        assert!(rendered.starts_with("doc_id = ") || rendered.contains("Utf8"));
    }

    #[test]
    fn filter_newline_is_safe_via_expr() {
        // Newlines in doc_ids are also safe — they're literal values, not SQL.
        let malicious = "doc1\n-- OR 1=1";
        let expr = build_single_expr(malicious);
        let rendered = format!("{}", expr);
        // The newline is part of the literal, not a SQL comment injection.
        assert!(rendered.contains("OR 1=1"));
    }

    #[test]
    fn filter_in_list_is_safe_via_expr() {
        // The IN-list path (build_doc_id_in_filter replacement) also uses Expr.
        let ids = ["evil\\' OR doc_id = 'admin".to_string()];
        let values: Vec<_> = ids.iter().map(|id| lit(id.as_str())).collect();
        let expr = lance::deps::datafusion::logical_expr::in_list(
            col("doc_id"),
            values,
            false,
        );
        let rendered = format!("{}", expr);
        // The entire malicious string is a single literal — no SQL injection.
        assert!(rendered.contains("admin"));
    }
}
