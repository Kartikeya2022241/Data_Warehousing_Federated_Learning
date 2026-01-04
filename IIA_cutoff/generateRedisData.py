import os
import random
from datetime import datetime, timedelta

import redis

# --- Config via env (matches your .env style) ---
REDIS_HOST = os.getenv("IIA_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("IIA_REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("IIA_REDIS_DB", "0"))

N_VITALS = int(os.getenv("IIA_REDIS_N_VITALS", "100"))
N_PRESCRIPTIONS = int(os.getenv("IIA_REDIS_N_PRESCRIPTIONS", "200"))
N_DRUGS = int(os.getenv("IIA_REDIS_N_DRUGS", "100"))
N_MANUFACTURERS = int(os.getenv("IIA_REDIS_N_MANUFACTURERS", "20"))

def _rand_date(days_back=120):
    d = datetime.now() - timedelta(days=random.randint(0, days_back))
    return d.strftime("%Y-%m-%d")

def load():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

    # -----------------
    # Manufacturers
    # -----------------
    for mid in range(1, N_MANUFACTURERS + 1):
        r.hset(
            f"Manufacturer:{mid}",
            mapping={
                "Manufacturer_ID": str(mid),
                "Name": f"Manufacturer {mid}",
                "Contact_Details": f"+1-555-{1000+mid}",
            },
        )

    # -----------------
    # Drugs
    # -----------------
    for did in range(1, N_DRUGS + 1):
        manufacturer = f"Manufacturer {random.randint(1, N_MANUFACTURERS)}"
        r.hset(
            f"Drug:{did}",
            mapping={
                "Drug_ID": str(did),
                "Drug_Name": f"Drug {did}",
                "Manufacturer": manufacturer,
                "Expiry_Date": _rand_date(days_back=365),
                "Batch_Number": f"BATCH-{random.randint(10000, 99999)}",
            },
        )

    # -----------------
    # Vitals (hash format)
    # Key pattern: Vitals:<patient_id>
    # -----------------
    for i in range(1, N_VITALS + 1):
        pid = i  # keep 1..100 for easy joining
        oxygen = random.choice([88, 90, 92, 94, 96, 98])
        r.hset(
            f"Vitals:{pid}",
            mapping={
                "patient_id": str(pid),
                "heart_rate": str(random.randint(55, 120)),
                "temperature": str(round(random.uniform(97.0, 103.0), 1)),
                "respiratory_rate": str(random.randint(10, 30)),
                "systolic": str(random.randint(90, 160)),
                "diastolic": str(random.randint(60, 100)),
                "oxygen_saturation": str(oxygen),
                "weight": str(round(random.uniform(45.0, 100.0), 1)),
            },
        )

    # -----------------
    # Prescriptions (hash format)
    # Key pattern: Prescription:<id>
    # -----------------
    for rx_id in range(1, N_PRESCRIPTIONS + 1):
        pid = random.randint(1, N_VITALS)         # joinable patient range
        did = random.randint(1, N_DRUGS)
        start = datetime.now() - timedelta(days=random.randint(1, 60))
        end = start + timedelta(days=random.randint(3, 14))
        r.hset(
            f"Prescription:{rx_id}",
            mapping={
                "prescription_id": str(rx_id),
                "patient_id": str(pid),
                "drug_id": str(did),
                "dosage": random.choice(["5mg", "10mg", "20mg", "1 tab"]),
                "frequency": random.choice(["OD", "BD", "TDS"]),
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
            },
        )

    print("Redis Pharma Records populated (manufacturers/drugs/vitals/prescriptions).")

def main():
    load()

if __name__ == "__main__":
    main()
