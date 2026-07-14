#!/usr/bin/env bash
# Creates one database per name listed in $POSTGRES_MULTIPLE_DATABASES
# (comma-separated) and installs the pgvector extension on the "app"
# database.
#
# The official postgres/pgvector images do NOT support
# POSTGRES_MULTIPLE_DATABASES natively -- that variable name is a
# convention, not a feature of the base image. This script is what
# actually reads it and does the work. It is mounted read-only at
# /docker-entrypoint-initdb.d/init-db.sh (see docker-compose.yml) and
# picked up automatically by the postgres entrypoint contract: any
# executable *.sh file in that directory is run once, only the first
# time the data directory is initialized.
#
# Single Postgres instance, three logical databases (app / litellm /
# langfuse) is a deliberate demo trade-off; see docs/adr/0004-single-postgres.md.

set -euo pipefail

if [ -z "${POSTGRES_MULTIPLE_DATABASES:-}" ]; then
  echo "init-db.sh: POSTGRES_MULTIPLE_DATABASES not set, nothing to do."
  exit 0
fi

# The postgres entrypoint creates one default database named after
# POSTGRES_DB, falling back to POSTGRES_USER when POSTGRES_DB is unset
# (our case). That default database is where we connect to run
# CREATE DATABASE for the extra ones.
ADMIN_DB="${POSTGRES_DB:-$POSTGRES_USER}"

create_database() {
  local db="$1"
  local exists
  exists=$(psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$ADMIN_DB" -tAc \
    "SELECT 1 FROM pg_database WHERE datname = '${db}'")
  if [ "$exists" = "1" ]; then
    echo "init-db.sh: database '${db}' already exists, skipping."
  else
    echo "init-db.sh: creating database '${db}'."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$ADMIN_DB" -c "CREATE DATABASE \"${db}\";"
  fi
}

IFS=',' read -ra DATABASES <<< "$POSTGRES_MULTIPLE_DATABASES"
for db in "${DATABASES[@]}"; do
  create_database "$db"
done

# Only "app" needs pgvector: it backs the LangGraph AsyncPostgresStore
# (long-term memory, HNSW index over embeddings). "litellm" and
# "langfuse" manage their own schemas via their own migrations.
echo "init-db.sh: ensuring pgvector extension on 'app'."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname app -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "init-db.sh: done."
