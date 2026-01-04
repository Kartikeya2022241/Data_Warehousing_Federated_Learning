import os
import re
import sqlite3
from typing import Any, Dict

import Main_run_file

DW_PATH = os.environ.get("IIA_DW_PATH", "data_warehouse.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DW_PATH, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


def _dw_has_tables() -> bool:
    try:
        conn = _connect()
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        conn.close()
        return len(tables) > 0
    except Exception:
        return False


def execute_sql2(sql: str) -> Dict[str, Any]:
    sql = (sql or "").strip()
    if not sql:
        raise ValueError("Empty SQL")

    conn = _connect()
    cur = conn.cursor()
    try:
        cur.execute(sql)

        if cur.description is not None:
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            return {"columns": cols, "rows": [list(r) for r in rows]}

        conn.commit()
        return {"columns": [], "rows": []}
    finally:
        conn.close()


def _extract_patient_id(text: str) -> int:
    m = re.search(r"patient[_\s-]*id\s*[:=]?\s*(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\bpredict\s+(\d+)\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0


def _llm_nl2sql(query: str) -> str:
    # Read env *at runtime*, not import time
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set (did you load .env before importing this module?)")

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed. Run: pip install openai") from e

    conn = _connect()
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

    schema_lines = []
    for (t,) in tables:
        cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
        schema_lines.append(f"- {t}({', '.join([c[1] for c in cols])})")
    conn.close()

    schema = "\n".join(schema_lines)

    system = (
        "You write SQLite SQL ONLY.\n"
        "Rules:\n"
        "1) Output ONLY the SQL query (no markdown, no explanation).\n"
        "2) Use ONLY tables/columns in the schema.\n"
        "3) Use LIMIT 50 unless user asks otherwise.\n"
        "4) SELECT/WITH only. No INSERT/UPDATE/DELETE/DROP.\n"
    )

    user = f"Schema:\n{schema}\n\nUser question:\n{query}\n"

    client = OpenAI(api_key=openai_key)
    resp = client.chat.completions.create(
        model=openai_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
    )

    sql = (resp.choices[0].message.content or "").strip()

    # Hard strip code fences if the model ever returns them
    if sql.startswith("```"):
        sql = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", sql).strip()

    if not sql.lower().startswith(("select", "with")):
        raise RuntimeError(f"LLM did not return a SELECT/WITH query. Got: {sql[:120]!r}")

    return sql


def get_query(query: str) -> str:
    q = (query or "").strip()
    if not q:
        raise ValueError("Empty query")

    # Ensure DW exists
    if not _dw_has_tables():
        Main_run_file.run_etl_and_train()

    s = q.lower()

    # Predict route
    q = (query or "").strip()
    if not q:
        raise ValueError("Empty query")

    if not _dw_has_tables():
        Main_run_file.run_etl_and_train()

    s = q.lower()

    # ✅ HARD ROUTE: any prediction intent => real-time inference (NOT SQL)
    pred_keywords = ["predict", "prediction", "risk", "probability", "chance", "likelihood"]
    if any(k in s for k in pred_keywords):
        pid = _extract_patient_id(q)
        if pid <= 0:
            return "Please provide a patient_id, e.g., 'predict risk for patient_id 77'."

        # real inference
        result = Main_run_file.predict_risk(pid)
        if not result.get("ok"):
            return f"Prediction failed: {result.get('error', 'unknown error')}"

        prob = result["risk_probability"]
        label = result["predicted_risk"]
        feats = result.get("features_used", {})
        clients = (result.get("meta") or {}).get("clients_used")

        # return a human-friendly message (your frontend can show this directly)
        return (
            f"Prediction for patient_id={pid}: risk_label={label} (prob={prob:.3f}). "
            f"Features used={feats}. Clients used={clients}."
        )

    # NL->SQL
    if os.getenv("OPENAI_API_KEY"):
        try:
            return _llm_nl2sql(q)
        except Exception as e:
            # Don't silently swallow: return a useful debug message
            err = str(e).replace("'", "''")
            return (
                "SELECT "
                f"'NL→SQL failed: {err} | "
                "Try: show patients | show billing | show claims | show vitals | "
                "risky patients | predict risk for patient_id 1' "
                "AS message;"
            )


    # Handwritten intents
    if "show tables" in s or "list tables" in s:
        return "SELECT name AS table_name FROM sqlite_master WHERE type='table' ORDER BY name;"

    if "show schema" in s:
        return (
            "SELECT m.name as table_name, p.name as column_name, p.type as column_type "
            "FROM sqlite_master m JOIN pragma_table_info(m.name) p "
            "WHERE m.type='table' ORDER BY m.name, p.cid;"
        )

    if "show patients" in s or "list patients" in s:
        return "SELECT patient_id, name, age, gender, source_system FROM warehouse_patients ORDER BY patient_id LIMIT 50;"

    if "show billing" in s or "list billing" in s:
        return "SELECT billing_id, patient_id, amount, date, source_system FROM warehouse_billing ORDER BY date LIMIT 50;"

    if "show claims" in s or "list claims" in s:
        return "SELECT claim_id, patient_id, amount, status, date, source_system FROM warehouse_claims ORDER BY date LIMIT 50;"

    if "show vitals" in s or "list vitals" in s:
        return (
            "SELECT patient_id, heart_rate, temperature, respiratory_rate, systolic, diastolic, "
            "oxygen_saturation, weight, source_system "
            "FROM warehouse_vitals LIMIT 50;"
        )

    if ("risky" in s or "high risk" in s) and ("patient" in s or "patients" in s):
        return (
            "SELECT patient_id, age, gender_male, total_billing, has_claim, total_claim_amount, oxygen_bin, risk_label "
            "FROM ml_patient_features "
            "WHERE risk_label = 1 "
            "ORDER BY total_billing DESC LIMIT 50;"
        )

    return (
        "SELECT 'Try: show patients | show billing | show claims | show vitals | risky patients | "
        "predict risk for patient_id 1' AS message;"
    )
