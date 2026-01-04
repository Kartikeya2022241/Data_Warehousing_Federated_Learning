# generate.py
from __future__ import annotations

import sys
import subprocess

import generateMongoDBData
import generateRedisData
import generatePostgresData
import generateSQLiteData


def _run(action: str, target: str):
    action = action.lower().strip()
    target = target.lower().strip()

    def call(mod):
        fn = getattr(mod, action, None)
        if fn is None:
            raise ValueError(f"{mod.__name__} has no function '{action}()'")
        fn()

    if target == "all":
        # Order: DB sources first, then file-based
        call(generateMongoDBData)
        call(generateRedisData)
        call(generatePostgresData)
        call(generateSQLiteData)
        return

    if target == "mongo":
        call(generateMongoDBData)
    elif target == "redis":
        call(generateRedisData)
    elif target == "postgres":
        call(generatePostgresData)
    elif target == "sqlite":
        call(generateSQLiteData)
    else:
        raise ValueError("target must be one of: all/mongo/redis/postgres/sqlite")


def main():
    if len(sys.argv) < 3:
        print("Usage: python generate.py <load|generate> <all|mongo|redis|postgres|sqlite>")
        sys.exit(1)

    action = sys.argv[1]
    target = sys.argv[2]
    _run(action, target)


if __name__ == "__main__":
    main()
