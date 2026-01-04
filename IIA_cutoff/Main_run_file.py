from __future__ import annotations

import os
import json
import time
import pickle
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


DW_PATH = os.environ.get("IIA_DW_PATH", "data_warehouse.db")
ARTIFACT_DIR = os.environ.get("IIA_ARTIFACT_DIR", "artifacts")
MODE = os.environ.get("IIA_MODE", "db").strip().lower()

MONGO_URI = os.environ.get("IIA_MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.environ.get("IIA_MONGO_DB", "MedicalRecords")
MONGO_PATIENTS_COLL = os.environ.get("IIA_MONGO_PATIENTS_COLL", "Patients")

REDIS_HOST = os.environ.get("IIA_REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("IIA_REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("IIA_REDIS_DB", "0"))
REDIS_VITALS_PREFIX = os.environ.get("IIA_REDIS_VITALS_PREFIX", "vitals:")

PG_HOST = os.environ.get("IIA_PG_HOST", "localhost")
PG_PORT = int(os.environ.get("IIA_PG_PORT", "5432"))
PG_USER = os.environ.get("IIA_PG_USER", "postgres")
PG_PASS = os.environ.get("IIA_PG_PASS", "admin")
PG_DB = os.environ.get("IIA_PG_DB", "postgres")

# accept both names; .env uses IIA_SQLITE_SOURCE_PATH
SQLITE_UPSTREAM_PATH = os.environ.get("IIA_SQLITE_SOURCE_PATH") or os.environ.get("IIA_SQLITE_DB_PATH", "BillingRecords.db")


BILL_THRESHOLD = float(os.environ.get("IIA_BILL_THRESHOLD", "600"))
MIN_CLIENT_ROWS = int(os.environ.get("IIA_MIN_CLIENT_ROWS", "5"))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def connect_dw(path: str = DW_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def df_to_sqlite(conn: sqlite3.Connection, df: pd.DataFrame, table: str, if_exists: str = "append") -> None:
    if df is None or df.empty:
        return
    df.to_sql(table, conn, if_exists=if_exists, index=False)

def safe_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def _now_year() -> int:
    from datetime import datetime
    return datetime.now().year

def _calc_age_from_dob_iso(dob_iso: str) -> int:
    try:
        y = int(str(dob_iso).split("-")[0])
        return max(0, _now_year() - y)
    except Exception:
        return 0


def create_dw_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS warehouse_patients (
            patient_id INTEGER,
            name TEXT,
            age INTEGER,
            gender TEXT,
            source_system TEXT,
            PRIMARY KEY (patient_id, source_system)
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS warehouse_billing (
            billing_id INTEGER,
            patient_id INTEGER,
            amount REAL,
            date TEXT,
            source_system TEXT,
            PRIMARY KEY (billing_id, source_system)
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS warehouse_claims (
            claim_id INTEGER,
            patient_id INTEGER,
            amount REAL,
            status TEXT,
            date TEXT,
            source_system TEXT,
            PRIMARY KEY (claim_id, source_system)
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS warehouse_vitals (
            patient_id INTEGER,
            heart_rate REAL,
            temperature REAL,
            respiratory_rate REAL,
            systolic REAL,
            diastolic REAL,
            oxygen_saturation REAL,
            weight REAL,
            source_system TEXT
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS warehouse_drugs (
            drug_id INTEGER,
            drug_name TEXT,
            manufacturer TEXT,
            expiry_date TEXT,
            batch_number TEXT,
            source_system TEXT,
            PRIMARY KEY (drug_id, source_system)
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS warehouse_manufacturers (
            manufacturer_id INTEGER,
            name TEXT,
            contact_details TEXT,
            source_system TEXT,
            PRIMARY KEY (manufacturer_id, source_system)
        );
    """)

    # ✅ prescriptions table (was zero because nothing loaded into it)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS warehouse_prescriptions (
            prescription_id INTEGER,
            patient_id INTEGER,
            drug_id INTEGER,
            dosage TEXT,
            frequency TEXT,
            start_date TEXT,
            end_date TEXT,
            source_system TEXT,
            PRIMARY KEY (prescription_id, source_system)
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_patient_features (
            patient_id INTEGER,
            age REAL,
            gender_male REAL,
            total_billing REAL,
            has_claim REAL,
            total_claim_amount REAL,
            oxygen_bin INTEGER,
            risk_label INTEGER
        );
    """)

    conn.commit()

def reset_tables(conn: sqlite3.Connection, tables: List[str]) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=OFF;")
    for t in tables:
        cur.execute(f"DELETE FROM {t};")
    cur.execute("PRAGMA foreign_keys=ON;")
    conn.commit()


def _load_from_mongo_patients() -> pd.DataFrame:
    from pymongo import MongoClient

    print(f"[MONGO] uri={MONGO_URI} db={MONGO_DB} coll={MONGO_PATIENTS_COLL}")
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    coll = db[MONGO_PATIENTS_COLL]

    rows = list(coll.find({}, {"_id": 0}))
    client.close()
    print(f"[MONGO] rows fetched={len(rows)}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    colmap = {c.lower(): c for c in df.columns}

    pid = colmap.get("patient_id") or colmap.get("patientid") or colmap.get("id")
    name = colmap.get("name")
    gender = colmap.get("gender")
    dob = colmap.get("date_of_birth") or colmap.get("dateofbirth") or colmap.get("dob")

    out = pd.DataFrame()
    out["patient_id"] = df[pid].apply(safe_int) if pid else np.arange(1, len(df) + 1)
    out["name"] = df[name].astype(str) if name else ""
    out["age"] = df[dob].astype(str).map(_calc_age_from_dob_iso).astype(int) if (dob and dob in df.columns) else 0
    out["gender"] = df[gender].astype(str) if gender else ""
    out["source_system"] = "mongo"

    before = len(out)
    out = out.sort_values("patient_id").drop_duplicates(subset=["patient_id", "source_system"], keep="first")
    after = len(out)
    if before != after:
        print(f"[MONGO] deduped patients: {before} -> {after} (duplicate Patient_IDs exist in Mongo)")
    return out

def _load_from_sqlite_billing() -> pd.DataFrame:
    print(f"[SQLITE-UPSTREAM] path={SQLITE_UPSTREAM_PATH}")
    if not os.path.exists(SQLITE_UPSTREAM_PATH):
        return pd.DataFrame()

    conn = sqlite3.connect(SQLITE_UPSTREAM_PATH)
    try:
        try:
            b = pd.read_sql_query("SELECT * FROM Billing;", conn)
        except Exception:
            b = pd.read_sql_query("SELECT * FROM billing;", conn)
    except Exception:
        b = pd.DataFrame()
    finally:
        conn.close()

    print(f"[SQLITE-UPSTREAM] billing rows fetched={len(b)}")
    if b.empty:
        return b

    colmap = {c.lower(): c for c in b.columns}
    bid = colmap.get("billing_id") or colmap.get("billingid") or colmap.get("invoice_number") or colmap.get("id")
    pid = colmap.get("patient_id") or colmap.get("patientid")
    amt = colmap.get("total_amount") or colmap.get("amount")
    dt = colmap.get("billing_date") or colmap.get("date")

    out = pd.DataFrame()
    out["billing_id"] = b[bid].apply(safe_int) if bid else np.arange(1, len(b) + 1)
    out["patient_id"] = b[pid].apply(safe_int) if pid else 0
    out["amount"] = b[amt].apply(safe_float) if amt else 0.0
    out["date"] = b[dt].astype(str) if dt else ""
    out["source_system"] = "sqlite"
    out = out.drop_duplicates(subset=["billing_id", "source_system"])
    return out

def _pg_connect():
    import psycopg2
    print(f"[PG] connect host={PG_HOST} port={PG_PORT} db={PG_DB} user={PG_USER}")
    return psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASS, dbname=PG_DB)

def _load_from_postgres_claims() -> pd.DataFrame:
    conn = _pg_connect()
    try:
        policies = pd.read_sql_query("SELECT policy_id, patient_id FROM policies;", conn)
        claims = pd.read_sql_query(
            "SELECT claim_id, policy_id, claim_amount, claim_status, claim_date FROM claims;",
            conn
        )
    finally:
        conn.close()

    print(f"[PG] policies rows fetched={len(policies)}")
    print(f"[PG] claims rows fetched={len(claims)}")

    if policies.empty or claims.empty:
        return pd.DataFrame()

    policies["policy_id"] = policies["policy_id"].apply(safe_int)
    policies["patient_id"] = policies["patient_id"].apply(safe_int)

    claims["claim_id"] = claims["claim_id"].apply(safe_int)
    claims["policy_id"] = claims["policy_id"].apply(safe_int)
    claims["amount"] = claims["claim_amount"].apply(safe_float)
    claims["status"] = claims["claim_status"].astype(str)
    claims["date"] = claims["claim_date"].astype(str)

    merged = claims.merge(policies, on="policy_id", how="inner")
    if merged.empty:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["claim_id"] = merged["claim_id"].apply(safe_int)
    out["patient_id"] = merged["patient_id"].apply(safe_int)
    out["amount"] = merged["amount"].apply(safe_float)
    out["status"] = merged["status"].astype(str)
    out["date"] = merged["date"].astype(str)
    out["source_system"] = "postgres"
    out = out.drop_duplicates(subset=["claim_id", "source_system"])
    return out

def _load_from_redis() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import redis

    print(f"[REDIS] connect host={REDIS_HOST} port={REDIS_PORT} db={REDIS_DB}")
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

    # ---- vitals ----
    vit_rows = []
    patterns = ["Vitals:*", "vitals:*", f"{REDIS_VITALS_PREFIX}*"]
    seen = set()

    for pat in patterns:
        for k in r.scan_iter(match=pat, count=1000):
            if k in seen:
                continue
            seen.add(k)
            h = r.hgetall(k) or {}
            if not h:
                continue
            pid = k.split(":", 1)[-1]
            h["_patient_id"] = pid
            vit_rows.append(h)

    vitals_df = pd.DataFrame(vit_rows)
    if not vitals_df.empty:
        def get_num(col: str) -> pd.Series:
            return vitals_df.get(col, pd.Series([0]*len(vitals_df))).apply(safe_float)

        out = pd.DataFrame()
        out["patient_id"] = vitals_df["_patient_id"].apply(safe_int)
        out["heart_rate"] = get_num("heart_rate")
        out["temperature"] = get_num("temperature")
        out["respiratory_rate"] = get_num("respiratory_rate")
        out["systolic"] = get_num("systolic")
        out["diastolic"] = get_num("diastolic")
        out["oxygen_saturation"] = get_num("oxygen_saturation")
        out["weight"] = get_num("weight")
        out["source_system"] = "redis"
        vitals_df = out
    else:
        vitals_df = pd.DataFrame()

    # ---- drugs ----
    drug_rows = []
    for k in r.scan_iter(match="Drug:*", count=1000):
        d = r.hgetall(k) or {}
        if d:
            drug_rows.append(d)
    drugs_df = pd.DataFrame(drug_rows)
    if not drugs_df.empty:
        out = pd.DataFrame()
        out["drug_id"] = drugs_df.get("Drug_ID", pd.Series(range(1, len(drugs_df)+1))).apply(safe_int)
        out["drug_name"] = drugs_df.get("Drug_Name", "").astype(str)
        out["manufacturer"] = drugs_df.get("Manufacturer", "").astype(str)
        out["expiry_date"] = drugs_df.get("Expiry_Date", "").astype(str)
        out["batch_number"] = drugs_df.get("Batch_Number", "").astype(str)
        out["source_system"] = "redis"
        out = out.drop_duplicates(subset=["drug_id", "source_system"])
        drugs_df = out
    else:
        drugs_df = pd.DataFrame()

    # ---- manufacturers ----
    manuf_rows = []
    for k in r.scan_iter(match="Manufacturer:*", count=1000):
        m = r.hgetall(k) or {}
        if m:
            manuf_rows.append(m)
    manufacturers_df = pd.DataFrame(manuf_rows)
    if not manufacturers_df.empty:
        out = pd.DataFrame()
        out["manufacturer_id"] = manufacturers_df.get("Manufacturer_ID", pd.Series(range(1, len(manufacturers_df)+1))).apply(safe_int)
        out["name"] = manufacturers_df.get("Name", "").astype(str)
        out["contact_details"] = manufacturers_df.get("Contact_Details", "").astype(str)
        out["source_system"] = "redis"
        out = out.drop_duplicates(subset=["manufacturer_id", "source_system"])
        manufacturers_df = out
    else:
        manufacturers_df = pd.DataFrame()

    # ✅ prescriptions (NEW)
    rx_rows = []
    for k in r.scan_iter(match="Prescription:*", count=1000):
        h = r.hgetall(k) or {}
        if h:
            rx_rows.append(h)
    prescriptions_df = pd.DataFrame(rx_rows)
    if not prescriptions_df.empty:
        out = pd.DataFrame()
        out["prescription_id"] = prescriptions_df.get("prescription_id", pd.Series(range(1, len(prescriptions_df)+1))).apply(safe_int)
        out["patient_id"] = prescriptions_df.get("patient_id", 0).apply(safe_int)
        out["drug_id"] = prescriptions_df.get("drug_id", 0).apply(safe_int)
        out["dosage"] = prescriptions_df.get("dosage", "").astype(str)
        out["frequency"] = prescriptions_df.get("frequency", "").astype(str)
        out["start_date"] = prescriptions_df.get("start_date", "").astype(str)
        out["end_date"] = prescriptions_df.get("end_date", "").astype(str)
        out["source_system"] = "redis"
        out = out.drop_duplicates(subset=["prescription_id", "source_system"])
        prescriptions_df = out
    else:
        prescriptions_df = pd.DataFrame()

    print(f"[REDIS] drugs={len(drugs_df)} manufacturers={len(manufacturers_df)} vitals={len(vitals_df)} prescriptions={len(prescriptions_df)}")
    return vitals_df, drugs_df, manufacturers_df, prescriptions_df


def db_mode_load(conn: sqlite3.Connection) -> None:
    patients_df = _load_from_mongo_patients()
    df_to_sqlite(conn, patients_df, "warehouse_patients")
    print(f"[DW] inserted warehouse_patients={len(patients_df)}")

    billing_df = _load_from_sqlite_billing()
    df_to_sqlite(conn, billing_df, "warehouse_billing")
    print(f"[DW] inserted warehouse_billing={len(billing_df)}")

    claims_df = _load_from_postgres_claims()
    df_to_sqlite(conn, claims_df, "warehouse_claims")
    print(f"[DW] inserted warehouse_claims={len(claims_df)}")

    vitals_df, drugs_df, manufacturers_df, prescriptions_df = _load_from_redis()
    df_to_sqlite(conn, vitals_df, "warehouse_vitals")
    df_to_sqlite(conn, drugs_df, "warehouse_drugs")
    df_to_sqlite(conn, manufacturers_df, "warehouse_manufacturers")
    df_to_sqlite(conn, prescriptions_df, "warehouse_prescriptions")
    print(f"[DW] inserted warehouse_prescriptions={len(prescriptions_df)}")


def _get_patient_features_merged(conn: sqlite3.Connection, patient_id: int) -> pd.DataFrame:
    """
    Build ONE merged feature row for prediction by combining all sources.
    If some source is missing, fill with defaults.
    """
    pid = safe_int(patient_id)
    if pid <= 0:
        return pd.DataFrame()

    # Patients (mongo)
    p = pd.read_sql_query(
        "SELECT patient_id, age, gender FROM warehouse_patients WHERE patient_id=? LIMIT 1;",
        conn,
        params=(pid,),
    )
    if p.empty:
        # No patient -> can't predict
        return pd.DataFrame()

    age = safe_float(p.loc[0, "age"], 0.0)
    g = str(p.loc[0, "gender"] or "").strip().lower()
    gender_male = 1.0 if g in {"m", "male"} else 0.0

    # Billing (sqlite)
    b = pd.read_sql_query(
        "SELECT COALESCE(SUM(amount),0) AS total_billing FROM warehouse_billing WHERE patient_id=?;",
        conn,
        params=(pid,),
    )
    total_billing = safe_float(b.loc[0, "total_billing"], 0.0) if not b.empty else 0.0

    # Claims (postgres)
    c = pd.read_sql_query(
        "SELECT COUNT(*) AS n_claims, COALESCE(SUM(amount),0) AS total_claim_amount "
        "FROM warehouse_claims WHERE patient_id=?;",
        conn,
        params=(pid,),
    )
    n_claims = safe_int(c.loc[0, "n_claims"], 0) if not c.empty else 0
    has_claim = 1.0 if n_claims > 0 else 0.0
    total_claim_amount = safe_float(c.loc[0, "total_claim_amount"], 0.0) if not c.empty else 0.0

    # Vitals (redis)
    v = pd.read_sql_query(
        "SELECT oxygen_saturation FROM warehouse_vitals WHERE patient_id=? LIMIT 1;",
        conn,
        params=(pid,),
    )
    oxygen_sat = safe_float(v.loc[0, "oxygen_saturation"], 0.0) if not v.empty else 0.0
    oxygen_bin = discretize_oxygen(oxygen_sat)

    # Risk label (same rule as training)
    risk_label = int(
        (total_billing > BILL_THRESHOLD)
        or (oxygen_bin == 0)
        or (total_claim_amount > BILL_THRESHOLD)
    )

    return pd.DataFrame([{
        "patient_id": pid,
        "age": age,
        "gender_male": gender_male,
        "total_billing": total_billing,
        "has_claim": has_claim,
        "total_claim_amount": total_claim_amount,
        "oxygen_bin": oxygen_bin,
        "risk_label": risk_label,
    }])


def _client_dataset(conn: sqlite3.Connection, client: str) -> pd.DataFrame:
    """
    Build a training dataset for ONE client (one source DB).
    Each client only 'sees' its own features; others are zero/default.
    This matches real FL intuition: local data is partial & siloed.
    """
    # We'll train on patients that exist in warehouse_patients (since Patient_ID is the join key).
    patients = pd.read_sql_query("SELECT patient_id, age, gender FROM warehouse_patients;", conn)
    if patients.empty:
        return pd.DataFrame()

    base = pd.DataFrame()
    base["patient_id"] = patients["patient_id"].apply(safe_int)
    base["age"] = patients["age"].apply(safe_float)
    base["gender_male"] = patients["gender"].astype(str).str.lower().map(lambda g: 1.0 if g in {"m","male"} else 0.0)

    # Defaults: “missing locally” = zeros / neutral bin
    base["total_billing"] = 0.0
    base["has_claim"] = 0.0
    base["total_claim_amount"] = 0.0
    base["oxygen_bin"] = 1  # neutral/unknown

    if client == "mongo":
        # mongo only provides age/gender (already there)
        pass

    elif client == "sqlite":
        billing = pd.read_sql_query(
            "SELECT patient_id, COALESCE(SUM(amount),0) AS total_billing "
            "FROM warehouse_billing GROUP BY patient_id;",
            conn,
        )
        if not billing.empty:
            billing["patient_id"] = billing["patient_id"].apply(safe_int)
            billing["total_billing"] = billing["total_billing"].apply(safe_float)
            base = base.merge(billing, on="patient_id", how="left", suffixes=("","_b"))
            base["total_billing"] = base["total_billing_b"].fillna(0.0)
            base.drop(columns=[c for c in base.columns if c.endswith("_b")], inplace=True)

    elif client == "postgres":
        claims = pd.read_sql_query(
            "SELECT patient_id, COUNT(*) AS n_claims, COALESCE(SUM(amount),0) AS total_claim_amount "
            "FROM warehouse_claims GROUP BY patient_id;",
            conn,
        )
        if not claims.empty:
            claims["patient_id"] = claims["patient_id"].apply(safe_int)
            claims["total_claim_amount"] = claims["total_claim_amount"].apply(safe_float)
            claims["has_claim"] = (claims["n_claims"].apply(safe_int) > 0).astype(float)

            # Merge safely: base already has has_claim/total_claim_amount defaults
            base = base.merge(
                claims[["patient_id", "has_claim", "total_claim_amount"]],
                on="patient_id",
                how="left",
                suffixes=("_base", "_pg"),
            )

            # Prefer postgres values when present, otherwise keep base defaults
            base["has_claim"] = base["has_claim_pg"].fillna(base["has_claim_base"]).astype(float)
            base["total_claim_amount"] = base["total_claim_amount_pg"].fillna(base["total_claim_amount_base"]).astype(float)

            base.drop(columns=["has_claim_base", "has_claim_pg", "total_claim_amount_base", "total_claim_amount_pg"], inplace=True)


    elif client == "redis":
        vit = pd.read_sql_query(
            "SELECT patient_id, oxygen_saturation FROM warehouse_vitals;",
            conn,
        )
        if not vit.empty:
            vit["patient_id"] = vit["patient_id"].apply(safe_int)
            vit["oxygen_saturation"] = vit["oxygen_saturation"].apply(safe_float)
            vit["oxygen_bin"] = vit["oxygen_saturation"].apply(discretize_oxygen)
            vit_agg = vit.groupby("patient_id", as_index=False)["oxygen_bin"].max()
            base = base.merge(vit_agg, on="patient_id", how="left", suffixes=("","_v"))
            base["oxygen_bin"] = base["oxygen_bin_v"].fillna(1).astype(int)
            base.drop(columns=[c for c in base.columns if c.endswith("_v")], inplace=True)

    else:
        raise ValueError(f"Unknown client: {client}")

    # Label is computed using the SAME rule (even if client doesn't observe all signals)
    base["risk_label"] = (
        (base["total_billing"] > BILL_THRESHOLD)
        | (base["oxygen_bin"] == 0)
        | (base["total_claim_amount"] > BILL_THRESHOLD)
    ).astype(int)

    return base



def discretize_oxygen(oxygen: float) -> int:
    if oxygen <= 0:
        return 1
    if oxygen < 92:
        return 0
    if oxygen < 96:
        return 1
    return 2

def build_ml_table(conn: sqlite3.Connection) -> None:
    reset_tables(conn, ["ml_patient_features"])

    patients = pd.read_sql_query("SELECT * FROM warehouse_patients;", conn)
    billing = pd.read_sql_query("SELECT * FROM warehouse_billing;", conn)
    claims = pd.read_sql_query("SELECT * FROM warehouse_claims;", conn)
    vitals = pd.read_sql_query("SELECT * FROM warehouse_vitals;", conn)

    if patients.empty:
        print("[ML] No patients -> ml_patient_features cleared.")
        return

    base = patients.copy()
    base["age"] = base["age"].apply(safe_float)
    base["gender_male"] = base["gender"].astype(str).str.lower().map(lambda g: 1.0 if g in {"m", "male"} else 0.0)

    bill_agg = (
        billing.assign(amount=billing["amount"].apply(safe_float))
        .groupby("patient_id", as_index=False)["amount"].sum()
        .rename(columns={"amount": "total_billing"})
        if not billing.empty
        else pd.DataFrame({"patient_id": base["patient_id"].unique(), "total_billing": 0.0})
    )

    if claims.empty:
        claim_agg = pd.DataFrame({"patient_id": base["patient_id"].unique(), "has_claim": 0.0, "total_claim_amount": 0.0})
    else:
        c = claims.copy()
        c["amount"] = c["amount"].apply(safe_float)
        claim_agg = (
            c.groupby("patient_id", as_index=False)
             .agg(has_claim=("claim_id", "size"), total_claim_amount=("amount", "sum"))
        )
        claim_agg["has_claim"] = (claim_agg["has_claim"] > 0).astype(float)

    if vitals.empty:
        vit_agg = pd.DataFrame({"patient_id": base["patient_id"].unique(), "oxygen_bin": 1})
    else:
        v = vitals.copy()
        v["oxygen_saturation"] = v["oxygen_saturation"].apply(safe_float)
        v["oxygen_bin"] = v["oxygen_saturation"].apply(discretize_oxygen)
        vit_agg = v.groupby("patient_id", as_index=False)["oxygen_bin"].max()

    feat = base.groupby("patient_id", as_index=False).agg(
        age=("age", "max"),
        gender_male=("gender_male", "max"),
    )
    feat = feat.merge(bill_agg, on="patient_id", how="left")
    feat = feat.merge(claim_agg, on="patient_id", how="left")
    feat = feat.merge(vit_agg, on="patient_id", how="left")

    feat["total_billing"] = feat["total_billing"].fillna(0.0).astype(float)
    feat["has_claim"] = feat["has_claim"].fillna(0.0).astype(float)
    feat["total_claim_amount"] = feat["total_claim_amount"].fillna(0.0).astype(float)
    feat["oxygen_bin"] = feat["oxygen_bin"].fillna(1).astype(int)

    feat["risk_label"] = (
        (feat["total_billing"] > BILL_THRESHOLD) |
        (feat["oxygen_bin"] == 0) |
        (feat["total_claim_amount"] > BILL_THRESHOLD)
    ).astype(int)

    df_to_sqlite(conn, feat[[
        "patient_id","age","gender_male","total_billing","has_claim","total_claim_amount","oxygen_bin","risk_label"
    ]], "ml_patient_features", if_exists="append")

    print(f"[ML] ml_patient_features built rows={len(feat)}")


@dataclass
class FLArtifacts:
    model_path: str
    scaler_path: str
    meta_path: str

def _feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df[["age", "gender_male", "total_billing", "has_claim", "total_claim_amount", "oxygen_bin"]].astype(np.float32).values
    y = df["risk_label"].astype(int).values
    return X, y


def _fit_local_lr(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """
    Fit a simple Logistic Regression model.
    Keep it deterministic and stable for small datasets.
    """
    # Important: need a solver that supports predict_proba reliably
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=500,
        random_state=42,
    )
    model.fit(X, y)
    return model


def _fedavg_coef_intercept(models: List[LogisticRegression], ns: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    FedAvg for scikit LR: weighted average of coef_ and intercept_ using sample counts ns.
    """
    if not models or not ns or len(models) != len(ns):
        raise ValueError("models and ns must be non-empty and same length")

    total = float(sum(ns))
    if total <= 0:
        raise ValueError("sum(ns) must be > 0")

    # ensure shapes align
    coef_shape = models[0].coef_.shape
    int_shape = models[0].intercept_.shape

    coef = np.zeros(coef_shape, dtype=np.float64)
    inter = np.zeros(int_shape, dtype=np.float64)

    for m, n in zip(models, ns):
        w = float(n) / total
        coef += w * m.coef_
        inter += w * m.intercept_

    return coef.astype(np.float32), inter.astype(np.float32)


def load_fl_artifacts() -> Tuple[LogisticRegression, StandardScaler, Dict[str, Any]]:
    """
    Load FedAvg model + scaler + meta from ARTIFACT_DIR.
    """
    model_path = os.path.join(ARTIFACT_DIR, "fedavg_lr.pkl")
    scaler_path = os.path.join(ARTIFACT_DIR, "fedavg_scaler.pkl")
    meta_path = os.path.join(ARTIFACT_DIR, "fedavg_meta.json")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(
            f"Missing FL artifacts. Expected:\n"
            f"- {model_path}\n- {scaler_path}\n- {meta_path}\n"
            f"Run ETL+train first (run_etl_and_train)."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, scaler, meta


def federated_train(conn: sqlite3.Connection) -> Optional[FLArtifacts]:
    """
    Train 4 local LogisticRegression models (mongo/sqlite/postgres/redis) and FedAvg them.
    """
    clients = ["mongo", "sqlite", "postgres", "redis"]

    local_models: List[LogisticRegression] = []
    ns: List[int] = []
    used: List[str] = []

    # We scale using the merged/global distribution (common in FL demos),
    # BUT we fit the scaler on merged rows to avoid per-client scaling mismatch.
    merged = pd.read_sql_query("SELECT patient_id FROM warehouse_patients;", conn)
    if merged.empty:
        print("[WARN] No patients; skipping training.")
        return None

    # Build merged training set (one row per patient) just for scaler fit
    merged_rows = []
    for pid in merged["patient_id"].tolist():
        row = _get_patient_features_merged(conn, pid)
        if not row.empty:
            merged_rows.append(row)
    if not merged_rows:
        print("[WARN] No merged rows; skipping training.")
        return None

    merged_df = pd.concat(merged_rows, ignore_index=True)
    X_all, y_all = _feature_matrix(merged_df)

    scaler = StandardScaler()
    Xs_all = scaler.fit_transform(X_all)

    # Helper to scale any client dataframe with the same scaler
    def _scale_client(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X, y = _feature_matrix(df)
        return scaler.transform(X), y

    # Train per client
    for c in clients:
        dfc = _client_dataset(conn, c)
        if dfc.empty:
            continue

        # Minimum rows + needs both classes
        if len(dfc) < MIN_CLIENT_ROWS or len(np.unique(dfc["risk_label"])) < 2:
            continue

        Xc, yc = _scale_client(dfc)
        m = _fit_local_lr(Xc, yc)

        local_models.append(m)
        ns.append(len(dfc))
        used.append(c)

    if not local_models:
        # fallback: train on merged_df so we at least produce a model
        print("[WARN] No valid clients; training global fallback.")
        global_model = _fit_local_lr(Xs_all, y_all)
        local_models = [global_model]
        ns = [len(merged_df)]
        used = ["global_fallback"]

    coef, inter = _fedavg_coef_intercept(local_models, ns)

    global_lr = LogisticRegression()
    global_lr.classes_ = np.array([0, 1])
    global_lr.coef_ = coef
    global_lr.intercept_ = inter
    global_lr.n_features_in_ = coef.shape[1]

    ensure_dir(ARTIFACT_DIR)
    model_path = os.path.join(ARTIFACT_DIR, "fedavg_lr.pkl")
    scaler_path = os.path.join(ARTIFACT_DIR, "fedavg_scaler.pkl")
    meta_path = os.path.join(ARTIFACT_DIR, "fedavg_meta.json")

    with open(model_path, "wb") as f:
        pickle.dump(global_lr, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "features": ["age","gender_male","total_billing","has_claim","total_claim_amount","oxygen_bin"],
        "label": "risk_label",
        "bill_threshold": BILL_THRESHOLD,
        "clients_used": used,
        "min_client_rows": MIN_CLIENT_ROWS,
        "n_rows_merged": int(len(merged_df)),
        "mode": MODE,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] FedAvg model saved. Clients used: {used}")
    return FLArtifacts(model_path=model_path, scaler_path=scaler_path, meta_path=meta_path)


def sanity_check(conn: sqlite3.Connection) -> None:
    tables = [
        "warehouse_patients",
        "warehouse_billing",
        "warehouse_claims",
        "warehouse_vitals",
        "warehouse_drugs",
        "warehouse_manufacturers",
        "warehouse_prescriptions",
        "ml_patient_features",
    ]
    out = {"db_path": os.path.abspath(DW_PATH), "counts": {}}
    for t in tables:
        out["counts"][t] = int(conn.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0])
    print("[SANITY]", json.dumps(out, indent=2))

def run_etl_and_train(reset_dw: bool = True) -> None:
    ensure_dir(ARTIFACT_DIR)
    if reset_dw and os.path.exists(DW_PATH):
        os.remove(DW_PATH)

    conn = connect_dw(DW_PATH)
    create_dw_schema(conn)

    reset_tables(conn, [
        "warehouse_patients",
        "warehouse_billing",
        "warehouse_claims",
        "warehouse_vitals",
        "warehouse_drugs",
        "warehouse_manufacturers",
        "warehouse_prescriptions",
        "ml_patient_features",
    ])

    if MODE != "db":
        raise RuntimeError("This version expects MODE=db")

    db_mode_load(conn)
    build_ml_table(conn)
    federated_train(conn)
    sanity_check(conn)
    conn.close()


def predict_risk(patient_id: int) -> Dict[str, Any]:
    patient_id = safe_int(patient_id)
    if patient_id <= 0:
        return {"ok": False, "error": "patient_id must be positive"}

    conn = connect_dw(DW_PATH)

    # Build merged row from all DBs
    df = _get_patient_features_merged(conn, patient_id)
    conn.close()

    if df.empty:
        return {"ok": False, "error": f"No data for patient_id={patient_id}. Run ETL first."}

    model, scaler, meta = load_fl_artifacts()

    X, _ = _feature_matrix(df)
    Xs = scaler.transform(X)
    prob = float(model.predict_proba(Xs)[0][1])
    pred = int(prob >= 0.5)

    return {
        "ok": True,
        "patient_id": patient_id,
        "predicted_risk": pred,
        "risk_probability": prob,
        "features_used": df.drop(columns=["risk_label"]).to_dict(orient="records")[0],
        "meta": {"clients_used": meta.get("clients_used"), "bill_threshold": meta.get("bill_threshold")},
    }


if __name__ == "__main__":
    run_etl_and_train(reset_dw=True)
