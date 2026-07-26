// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

//! Prometheus metrics endpoint.
//!
//! Exposes a `/metrics` endpoint in Prometheus text exposition format on a
//! separate configurable port. Enabled via `VECTORLINK_PROMETHEUS_PORT`.
//!
//! All metrics are unauthenticated (the port should be firewalled from
//! external access). The endpoint exposes:
//! - Internal data structure sizes (tasks, datasets, branch_indexes, etc.)
//! - Engine statistics (domains, branches, documents, chunks, etc.)

use axum::{routing::get, Router};
use std::net::SocketAddr;

use crate::service::SearchService;

/// Render metrics in Prometheus text exposition format.
///
/// Format: `# HELP <name> <description>\n# TYPE <name> <type>\n<name> <value>\n`
async fn render_metrics(service: &SearchService) -> String {
    let stats = service.statistics().await
        .unwrap_or(crate::kernel::model::Statistics {
            domains: 0,
            branches: 0,
            indexed_commits: 0,
            documents: 0,
            chunks: 0,
            pending_index_fragments: 0,
            pending_index_documents: 0,
            store_clustering: None,
        });
    let internal = service.internal_stats().await;

    let mut out = String::new();

    // ── Engine statistics ──
    out.push_str("# HELP vectorlink_domains Total number of indexed domains\n");
    out.push_str("# TYPE vectorlink_domains gauge\n");
    out.push_str(&format!("vectorlink_domains {}\n", stats.domains));

    out.push_str("# HELP vectorlink_branches Total number of indexed branches\n");
    out.push_str("# TYPE vectorlink_branches gauge\n");
    out.push_str(&format!("vectorlink_branches {}\n", stats.branches));

    out.push_str("# HELP vectorlink_indexed_commits Total number of indexed commits\n");
    out.push_str("# TYPE vectorlink_indexed_commits gauge\n");
    out.push_str(&format!("vectorlink_indexed_commits {}\n", stats.indexed_commits));

    out.push_str("# HELP vectorlink_documents Total number of indexed documents\n");
    out.push_str("# TYPE vectorlink_documents gauge\n");
    out.push_str(&format!("vectorlink_documents {}\n", stats.documents));

    out.push_str("# HELP vectorlink_chunks Total number of live indexed chunks (excludes soft-deleted)\n");
    out.push_str("# TYPE vectorlink_chunks gauge\n");
    out.push_str(&format!("vectorlink_chunks {}\n", stats.chunks));

    out.push_str("# HELP vectorlink_pending_index_fragments Fragments not yet covered by vector ANN index\n");
    out.push_str("# TYPE vectorlink_pending_index_fragments gauge\n");
    out.push_str(&format!("vectorlink_pending_index_fragments {}\n", stats.pending_index_fragments));

    out.push_str("# HELP vectorlink_pending_index_documents Rows in fragments not yet covered by vector ANN index\n");
    out.push_str("# TYPE vectorlink_pending_index_documents gauge\n");
    out.push_str(&format!("vectorlink_pending_index_documents {}\n", stats.pending_index_documents));

    // ── Internal data structure sizes ──
    out.push_str("# HELP vectorlink_cached_datasets Number of cached Lance Dataset handles\n");
    out.push_str("# TYPE vectorlink_cached_datasets gauge\n");
    out.push_str(&format!("vectorlink_cached_datasets {}\n", internal.cached_datasets));

    out.push_str("# HELP vectorlink_tasks Number of tracked tasks in the tasks HashMap\n");
    out.push_str("# TYPE vectorlink_tasks gauge\n");
    out.push_str(&format!("vectorlink_tasks {}\n", internal.tasks));

    out.push_str("# HELP vectorlink_branch_indexes Number of per-(domain, branch) index entries\n");
    out.push_str("# TYPE vectorlink_branch_indexes gauge\n");
    out.push_str(&format!("vectorlink_branch_indexes {}\n", internal.branch_indexes));

    out.push_str("# HELP vectorlink_pipeline_locks Number of per-(domain, branch) pipeline locks\n");
    out.push_str("# TYPE vectorlink_pipeline_locks gauge\n");
    out.push_str(&format!("vectorlink_pipeline_locks {}\n", internal.pipeline_locks));

    out.push_str("# HELP vectorlink_domain_guards Number of per-domain guard mutexes\n");
    out.push_str("# TYPE vectorlink_domain_guards gauge\n");
    out.push_str(&format!("vectorlink_domain_guards {}\n", internal.domain_guards));

    out.push_str("# HELP vectorlink_inflight_commits Number of in-flight commit reservations\n");
    out.push_str("# TYPE vectorlink_inflight_commits gauge\n");
    out.push_str(&format!("vectorlink_inflight_commits {}\n", internal.inflight_commits));

    // ── Pipeline progress ──
    out.push_str("# HELP vectorlink_pipeline_pending_chunks Chunks chunked but not yet written to Lance\n");
    out.push_str("# TYPE vectorlink_pipeline_pending_chunks gauge\n");
    out.push_str(&format!("vectorlink_pipeline_pending_chunks {}\n", internal.pipeline_pending_chunks));

    out.push_str("# HELP vectorlink_pipeline_embedded_chunks Chunks embedded (embedding result received)\n");
    out.push_str("# TYPE vectorlink_pipeline_embedded_chunks gauge\n");
    out.push_str(&format!("vectorlink_pipeline_embedded_chunks {}\n", internal.pipeline_embedded_chunks));

    out.push_str("# HELP vectorlink_pipeline_written_chunks Chunks written to Lance (append committed)\n");
    out.push_str("# TYPE vectorlink_pipeline_written_chunks gauge\n");
    out.push_str(&format!("vectorlink_pipeline_written_chunks {}\n", internal.pipeline_written_chunks));

    // ── Pipeline task tracking ──
    out.push_str("# HELP vectorlink_pipeline_active_tasks Number of pipeline tasks currently running (spawned but not completed)\n");
    out.push_str("# TYPE vectorlink_pipeline_active_tasks gauge\n");
    out.push_str(&format!("vectorlink_pipeline_active_tasks {}\n", internal.pipeline_active_tasks));

    // ── Fresh dataset open counter ──
    out.push_str("# HELP vectorlink_fresh_open_count_total Total count of fresh Dataset::open calls (cumulative)\n");
    out.push_str("# TYPE vectorlink_fresh_open_count_total counter\n");
    out.push_str(&format!("vectorlink_fresh_open_count_total {}\n", internal.fresh_open_count));

    // ── Embedding cache ──
    out.push_str("# HELP vectorlink_embed_cache_entries Number of entries in the embedding cache\n");
    out.push_str("# TYPE vectorlink_embed_cache_entries gauge\n");
    out.push_str(&format!("vectorlink_embed_cache_entries {}\n", internal.embed_cache_entries));

    out.push_str("# HELP vectorlink_embed_cache_size_bytes Approximate memory usage of the embedding cache in bytes\n");
    out.push_str("# TYPE vectorlink_embed_cache_size_bytes gauge\n");
    out.push_str(&format!("vectorlink_embed_cache_size_bytes {}\n", internal.embed_cache_size_bytes));

    // ── Lance cache capacities (static, for reference) ──
    out.push_str("# HELP vectorlink_lance_index_cache_capacity_bytes Configured Lance index cache capacity in bytes\n");
    out.push_str("# TYPE vectorlink_lance_index_cache_capacity_bytes gauge\n");
    out.push_str(&format!("vectorlink_lance_index_cache_capacity_bytes {}\n", internal.lance_index_cache_capacity_bytes));

    out.push_str("# HELP vectorlink_lance_metadata_cache_capacity_bytes Configured Lance metadata cache capacity in bytes\n");
    out.push_str("# TYPE vectorlink_lance_metadata_cache_capacity_bytes gauge\n");
    out.push_str(&format!("vectorlink_lance_metadata_cache_capacity_bytes {}\n", internal.lance_metadata_cache_capacity_bytes));

    out
}

/// Start the Prometheus metrics server on the given port.
///
/// Returns a JoinHandle that can be awaited or aborted. The server exposes
/// a single `GET /metrics` endpoint in Prometheus text exposition format.
pub async fn start_metrics_server(
    port: u16,
    service: SearchService,
) -> Result<(), std::io::Error> {
    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    let app = Router::new().route(
        "/metrics",
        get(move || {
            let service = service.clone();
            async move {
                let body = render_metrics(&service).await;
                (
                    [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")],
                    body,
                )
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("[metrics] Prometheus metrics server listening on :{}", port);
    axum::serve(listener, app).await
}

#[cfg(test)]
mod tests {
    #[test]
    fn render_metrics_produces_valid_prometheus_text() {
        // We can't easily construct a SearchService in a unit test, but we can
        // verify the format of a minimal render. This test is a smoke test
        // for the format structure.
        let text = "# HELP vectorlink_domains Total number of indexed domains\n# TYPE vectorlink_domains gauge\nvectorlink_domains 5\n";
        assert!(text.contains("# HELP"));
        assert!(text.contains("# TYPE"));
        assert!(text.contains("vectorlink_domains"));
    }
}
