from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import os
import subprocess
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    raise RuntimeError("Missing dependency: pip install python-dotenv")

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# quick debug (DO NOT print the key)
print("[ENV] .env path:", ENV_PATH)
print("[ENV] OPENAI_API_KEY set?", bool(os.getenv("OPENAI_API_KEY")))
print("[ENV] OPENAI_MODEL:", os.getenv("OPENAI_MODEL"))

import Query_related
import Main_run_file


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"ok": True})

    @app.route("/sanity", methods=["GET"])
    def sanity():
        # Returns row counts from DW tables
        conn = Main_run_file.connect_dw(Main_run_file.DW_PATH)
        try:
            out = Main_run_file.sanity_check(conn)
            return jsonify({"ok": True, "sanity": out})
        finally:
            conn.close()

    @app.route("/getSQL", methods=["POST"])
    def get_sql():
        payload = request.get_json(silent=True) or {}
        query = (payload.get("query") or "").strip()
        if not query:
            return jsonify({"ok": False, "error": "Missing 'query'"}), 400
        if "predict" in query.lower():
            return jsonify({"ok": False, "error": "Use /predict endpoint for prediction queries."}), 400

        try:
            sql = Query_related.get_query(query)
            # Keep compatibility with frontend expecting sql/result fields
            return jsonify({"ok": True, "query": query, "sql": sql, "result": sql})
        except Exception as e:
            return jsonify({"ok": False, "query": query, "error": str(e)}), 500

    @app.route("/exSQL", methods=["POST"])
    def execute_sql():
        payload = request.get_json(silent=True) or {}
        sql = (payload.get("query") or "").strip()
        if not sql:
            return jsonify({"ok": False, "error": "Missing 'query' (SQL)"}), 400
        try:
            result = Query_related.execute_sql2(sql)
            return jsonify({"ok": True, "query": sql, "results": result})
        except Exception as e:
            return jsonify({"ok": False, "query": sql, "error": str(e)}), 500

    @app.route("/predict", methods=["POST"])
    def predict():
        payload = request.get_json(silent=True) or {}
        query = (payload.get("query") or "").strip()
        if not query:
            return jsonify({"ok": False, "error": "Missing 'query'"}), 400
        if "predict" not in query.lower():
            return jsonify({"ok": False, "error": "Prediction query must include the word 'predict'."}), 400

        try:
            result = Query_related.get_query(query)  # returns text answer
            return jsonify({"ok": True, "query": query, "result": result})
        except Exception as e:
            return jsonify({"ok": False, "query": query, "error": str(e)}), 500

    return app


def _run_generators_if_enabled():
    """
    Runs: python generate.py load all
    using an absolute path + correct cwd.
    """
    if os.environ.get("IIA_RUN_GENERATORS", "1") != "1":
        print("[INIT] Generators disabled (IIA_RUN_GENERATORS != 1)")
        return

    here = os.path.dirname(os.path.abspath(__file__))
    gen_path = os.path.join(here, "generate.py")

    if not os.path.exists(gen_path):
        print(f"[WARN] generate.py not found at: {gen_path} (skipping generator step)")
        return

    print("[INIT] Running generators: python generate.py load all")
    try:
        cp = subprocess.run(
            ["python3", gen_path, "load", "all"],
            cwd=here,
            capture_output=True,
            text=True,
            check=False,
        )
        if cp.stdout.strip():
            print("[GEN:STDOUT]\n" + cp.stdout)
        if cp.stderr.strip():
            print("[GEN:STDERR]\n" + cp.stderr)

        if cp.returncode != 0:
            print(f"[WARN] Generator returned non-zero exit code: {cp.returncode}")
        else:
            print("[OK] Generator step finished.")
    except Exception as e:
        print(f"[WARN] Generator step failed: {e}")


def init_pipeline_once():
    """
    1) populate upstream DBs via generate.py load all (optional)
    2) create/refresh DW and run ETL + FedAvg training
    """
    _run_generators_if_enabled()

    reset_dw = (os.environ.get("IIA_RESET_DW", "1") == "1")
    print(f"[INIT] Running ETL + FedAvg train (reset_dw={reset_dw})")
    Main_run_file.run_etl_and_train(reset_dw=reset_dw)


def start_etl_thread():
    interval = int(os.environ.get("IIA_ETL_INTERVAL_SECONDS", "60"))
    train_every = int(os.environ.get("IIA_TRAIN_EVERY_N_ETL", "5"))
    reset_first = (os.environ.get("IIA_RESET_DW_FIRST_IN_THREAD", "0") == "1")

    t = threading.Thread(
        target=Main_run_file.run_etl_periodically,
        kwargs={
            "etl_interval_seconds": interval,
            "train_every_n_runs": train_every,
            "reset_dw_first": reset_first,
        },
        daemon=True,
    )
    t.start()
    return t


if __name__ == "__main__":
    app = create_app()

    # ✅ ALWAYS init once because you already run with use_reloader=False
    init_pipeline_once()

    if os.environ.get("IIA_START_ETL", "0") == "1":
        start_etl_thread()

    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5000")),
        debug=(os.environ.get("FLASK_DEBUG", "1") == "1"),
        use_reloader=False,
    )
