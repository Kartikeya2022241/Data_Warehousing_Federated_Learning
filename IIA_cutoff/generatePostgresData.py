import random
from datetime import datetime, timedelta

import psycopg2


def load(
    host="localhost",
    port=5432,
    dbname="postgres",
    user="postgres",
    password="admin",
    n_providers=10,
    n_policies=100,
    n_claims=200,
):
    """
    HARD RESET + populate Postgres tables:
      - insurance_providers
      - policies
      - claims

    Why HARD RESET?
    Your existing tables were created with different column names (old schema),
    so INSERTs like (provider_name, provider_contact) fail.
    Dropping & recreating guarantees schema matches.
    """

    conn = psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)
    conn.autocommit = False
    cur = conn.cursor()

    # ---- Drop old schema (this fixes your provider_name missing error) ----
    cur.execute("DROP TABLE IF EXISTS claims CASCADE;")
    cur.execute("DROP TABLE IF EXISTS policies CASCADE;")
    cur.execute("DROP TABLE IF EXISTS insurance_providers CASCADE;")

    # ---- Create fresh schema (lowercase identifiers) ----
    cur.execute(
        """
        CREATE TABLE insurance_providers (
            insurance_provider_id SERIAL PRIMARY KEY,
            provider_name         VARCHAR(100) NOT NULL,
            provider_contact      VARCHAR(100) NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE policies (
            policy_id             SERIAL PRIMARY KEY,
            policy_number         VARCHAR(50) NOT NULL,
            patient_id            INT NOT NULL,
            insurance_provider_id INT NOT NULL REFERENCES insurance_providers(insurance_provider_id),
            coverage_amount       NUMERIC(12,2) NOT NULL,
            start_date            DATE NOT NULL,
            end_date              DATE NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE claims (
            claim_id      SERIAL PRIMARY KEY,
            policy_id     INT NOT NULL REFERENCES policies(policy_id),
            billing_id    INT,
            claim_amount  NUMERIC(12,2) NOT NULL,
            claim_status  VARCHAR(30) NOT NULL,
            claim_date    DATE NOT NULL
        );
        """
    )

    # ---- Insert providers ----
    providers = [
        (f"Provider_{i+1}", f"+1-555-{random.randint(1000,9999)}")
        for i in range(n_providers)
    ]
    cur.executemany(
        "INSERT INTO insurance_providers (provider_name, provider_contact) VALUES (%s, %s);",
        providers,
    )

    cur.execute("SELECT insurance_provider_id FROM insurance_providers ORDER BY insurance_provider_id;")
    provider_ids = [r[0] for r in cur.fetchall()]

    # ---- Insert policies ----
    base_date = datetime(2023, 1, 1)
    policies = []
    for i in range(n_policies):
        patient_id = random.randint(1, 100)     # aligns with your deduped DW patient_ids
        prov_id = random.choice(provider_ids)
        start = base_date + timedelta(days=random.randint(0, 365))
        end = start + timedelta(days=random.randint(180, 365))
        policies.append(
            (
                f"POL-{100000+i}",
                patient_id,
                prov_id,
                round(random.uniform(5000, 200000), 2),
                start.date(),
                end.date(),
            )
        )

    cur.executemany(
        """
        INSERT INTO policies
          (policy_number, patient_id, insurance_provider_id, coverage_amount, start_date, end_date)
        VALUES (%s, %s, %s, %s, %s, %s);
        """,
        policies,
    )

    cur.execute("SELECT policy_id FROM policies ORDER BY policy_id;")
    policy_ids = [r[0] for r in cur.fetchall()]

    # ---- Insert claims (THIS is what you were missing before) ----
    statuses = ["Approved", "Rejected", "Pending"]
    claims = []
    for _ in range(n_claims):
        pol_id = random.choice(policy_ids)
        billing_id = random.randint(1, 100)     # matches your SQLite billing generator range
        amt = round(random.uniform(50, 5000), 2)
        st = random.choice(statuses)
        dt = base_date + timedelta(days=random.randint(0, 730))
        claims.append((pol_id, billing_id, amt, st, dt.date()))

    cur.executemany(
        """
        INSERT INTO claims (policy_id, billing_id, claim_amount, claim_status, claim_date)
        VALUES (%s, %s, %s, %s, %s);
        """,
        claims,
    )

    conn.commit()
    cur.close()
    conn.close()

    print("PostgreSQL Insurance Records populated (providers/policies/claims).")


if __name__ == "__main__":
    load()
