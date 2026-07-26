// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

#![forbid(unsafe_code)]

//! Ingest — NDJSON push stream parsing.
//!
//! Incremental line-by-line parsing of the push body into Operations.
//! Pure parse helper separately tested; fails loud on malformed lines.

use crate::kernel::error::IngestError;
use crate::kernel::model::Operation;

/// Parse a single NDJSON line into an Operation.
/// Pure, unit-tested. Fails loud on malformed input.
pub fn parse_operation_line(line: &str, line_num: usize) -> Result<Operation, IngestError> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Err(IngestError::MalformedLine {
            line: line_num,
            detail: "empty line".to_owned(),
        });
    }
    serde_json::from_str::<Operation>(trimmed).map_err(|e| IngestError::MalformedLine {
        line: line_num,
        detail: e.to_string(),
    })
}

/// Parse an entire NDJSON body into a vector of Operations.
/// Skips blank lines. Fails loud on the first malformed non-blank line
/// (systemic failure — malformed input halts the entire push).
pub fn parse_ndjson_body(body: &str) -> Result<Vec<Operation>, IngestError> {
    let mut operations = Vec::new();
    for (idx, line) in body.lines().enumerate() {
        let line_num = idx + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let op = parse_operation_line(trimmed, line_num)?;
        operations.push(op);
    }
    Ok(operations)
}

/// Extract the doc_type from a document IRI.
/// Convention: the IRI is like "terminusdb:///star-wars/People/20",
/// where the type is the segment before the last one ("People").
/// For plain "Type/id" format, the type is everything before the last '/'.
pub fn extract_doc_type(doc_id: &str) -> String {
    // Try to extract from a full IRI: last two path segments are Type/Id.
    if let Some(after_scheme) = doc_id.find("///") {
        let path = &doc_id[after_scheme + 3..];
        let segments: Vec<&str> = path.split('/').collect();
        if segments.len() >= 2 {
            return segments[segments.len() - 2].to_owned();
        }
    }
    // Fallback: split on '/' and take the second-to-last segment.
    let segments: Vec<&str> = doc_id.split('/').collect();
    if segments.len() >= 2 {
        segments[segments.len() - 2].to_owned()
    } else {
        "unknown".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_inserted_operation() {
        let line = r#"{"op":"Inserted","id":"terminusdb:///star-wars/People/20","string":"Yoda is wise."}"#;
        let op = parse_operation_line(line, 1).unwrap();
        match op {
            Operation::Inserted { id, string } => {
                assert_eq!(id, "terminusdb:///star-wars/People/20");
                assert_eq!(string, "Yoda is wise.");
            }
            _ => panic!("expected Inserted"),
        }
    }

    #[test]
    fn parse_changed_operation() {
        let line = r#"{"op":"Changed","id":"doc/1","string":"new content"}"#;
        let op = parse_operation_line(line, 1).unwrap();
        assert!(matches!(op, Operation::Changed { .. }));
    }

    #[test]
    fn parse_deleted_operation() {
        let line = r#"{"op":"Deleted","id":"doc/1"}"#;
        let op = parse_operation_line(line, 1).unwrap();
        assert!(matches!(op, Operation::Deleted { .. }));
    }

    #[test]
    fn parse_error_operation() {
        let line = r#"{"op":"Error","message":"render failed"}"#;
        let op = parse_operation_line(line, 1).unwrap();
        assert!(matches!(op, Operation::Error { .. }));
    }

    #[test]
    fn parse_abort_operation() {
        let line = r#"{"op":"Abort"}"#;
        let op = parse_operation_line(line, 1).unwrap();
        assert!(matches!(op, Operation::Abort));
    }

    #[test]
    fn parse_malformed_line_fails_loud() {
        let line = "not json at all";
        let result = parse_operation_line(line, 5);
        assert!(result.is_err());
        match result.unwrap_err() {
            IngestError::MalformedLine { line, detail } => {
                assert_eq!(line, 5);
                assert!(!detail.is_empty());
            }
            _ => panic!("expected MalformedLine"),
        }
    }

    #[test]
    fn parse_ndjson_body_skips_blank_lines() {
        let body = concat!(
            r#"{"op":"Inserted","id":"doc/1","string":"hello"}"#,
            "\n\n",
            r#"{"op":"Deleted","id":"doc/2"}"#,
            "\n"
        );
        let ops = parse_ndjson_body(body).unwrap();
        assert_eq!(ops.len(), 2);
    }

    #[test]
    fn parse_ndjson_body_fails_on_malformed_line() {
        let body = concat!(
            r#"{"op":"Inserted","id":"doc/1","string":"hello"}"#,
            "\n",
            "bad line\n",
            r#"{"op":"Deleted","id":"doc/2"}"#,
        );
        let result = parse_ndjson_body(body);
        assert!(result.is_err());
    }

    #[test]
    fn extract_doc_type_from_terminusdb_iri() {
        assert_eq!(
            extract_doc_type("terminusdb:///star-wars/People/20"),
            "People"
        );
    }

    #[test]
    fn extract_doc_type_from_simple_path() {
        assert_eq!(extract_doc_type("Species/8"), "Species");
    }

    #[test]
    fn extract_doc_type_single_segment() {
        assert_eq!(extract_doc_type("something"), "unknown");
    }
}
