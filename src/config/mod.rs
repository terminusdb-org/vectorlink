#![forbid(unsafe_code)]

//! Configuration — resolve CLI + environment variables.
//! Pure validation with one effectful entry point for env loading.

/// Application configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub admin_user: String,
    pub admin_secret: String,
    pub port: u16,
}

impl Config {
    /// Load configuration from environment variables with defaults.
    pub fn from_env() -> Self {
        Self {
            admin_user: std::env::var("TDB_SEARCH_ADMIN_USER")
                .unwrap_or_else(|_| "admin".to_owned()),
            admin_secret: std::env::var("TDB_SEARCH_ADMIN_SECRET")
                .unwrap_or_else(|_| "root".to_owned()),
            port: std::env::var("TDB_SEARCH_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(8080),
        }
    }

    /// Create a config with explicit values (used in tests).
    pub fn new(admin_user: String, admin_secret: String, port: u16) -> Self {
        Self { admin_user, admin_secret, port }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            admin_user: "admin".to_owned(),
            admin_secret: "root".to_owned(),
            port: 8080,
        }
    }
}
