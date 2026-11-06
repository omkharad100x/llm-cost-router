"""Explicit local database initialization and read-only verification."""
import argparse
import json
from database import DatabaseError, initialize_database, verify_database
from database_config import load_database_config

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("initialize", "verify"))
    args = parser.parse_args(argv)
    try:
        config = load_database_config()
        if args.action == "initialize":
            initialize_database(config)
        print(json.dumps(verify_database(config)))
        return 0
    except (DatabaseError, ValueError):
        print(json.dumps({"connection": "Unavailable", "message": "Check database configuration, schema and storage access."}))
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
