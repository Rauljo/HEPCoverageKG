import logging
from hepcoveragekg.aliases.store import connect
from hepcoveragekg.aliases.run import build

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    conn = connect()
    results = build(conn)
    print("Alias Pipeline Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
