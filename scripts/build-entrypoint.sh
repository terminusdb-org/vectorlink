#!/bin/sh
# Entrypoint for the tdb-search build container.
# Ensures HOME exists (required by cargo/rustup even when running as an
# arbitrary --user uid:gid), then exec's the user's command.
# The image pre-creates /tmp/build-home with mode 777, so this is a no-op
# in the common case — it's here as a safety net for custom HOME values.

# WHY: mkdir may emit "File exists" to stderr when HOME already exists (harmless).
# INVARIANT: the directory either already exists (image layer) or is created now;
# either way exec proceeds with a valid HOME.
# CONSEQUENCE: if mkdir truly fails (read-only fs), cargo will error with a clear
# message about HOME — the suppression hides only the benign "exists" case.
mkdir -p "${HOME:-/tmp/build-home}" 2>/dev/null
exec "$@"
