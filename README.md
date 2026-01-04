
# 🏥 Intelligent Insurance Analytics (IIA)
**Federated Learning–based Healthcare Risk Prediction System**

---

## 📌 Overview

This project is an end-to-end data engineering and machine learning system that:

- Ingests healthcare data from **multiple heterogeneous databases**
- Builds a **central data warehouse**
- Trains **federated machine learning models** across data silos
- Aggregates models using **Federated Averaging (FedAvg)**
- Performs **real-time patient risk prediction**
- Supports **natural-language queries (NL → SQL)** for analytics

> ⚠️ **Important**  
> Predictions are performed **in real time** using the trained federated model.  
> Stored labels are used **only for training**, not for inference.

---

## 🧠 System Architecture

```

MongoDB        PostgreSQL        Redis          SQLite
(Patients)     (Claims)         (Vitals)       (Billing)
│              │               │               │
└────────────── ETL + Standardization ─────────┘
│
SQLite Data Warehouse
│
Federated Learning (4 Clients)
│
FedAvg Global Model
│
Real-Time Risk Prediction API

```

---

## 🗂️ Data Sources

| Source | Database | Data |
|------|----------|------|
| MongoDB | MedicalRecords | Patient demographics |
| SQLite | BillingRecords.db | Billing & invoices |
| PostgreSQL | postgres | Insurance policies & claims |
| Redis | Key–Value Store | Vitals, drugs, prescriptions |

Each data source behaves as an **independent federated client**.

---

## 🧬 Federated Learning Logic

This project implements **true horizontal federated learning**.

### Federated Clients
- **MongoDB client** → Age, gender  
- **SQLite client** → Billing totals  
- **PostgreSQL client** → Claims & insurance data  
- **Redis client** → Vitals (oxygen saturation)

Each client:
- Trains a **local Logistic Regression model**
- Uses **only its own data**
- Never shares raw data

### Risk Label Rule (Training Only)

```

High risk if:
total_billing > threshold
OR oxygen_saturation < 92
OR total_claim_amount > threshold

```

---

### Federated Averaging (FedAvg)

Local models are aggregated using:

```

GlobalWeight = Σ (nᵢ / N) × LocalWeightᵢ

```

Artifacts produced:
```

artifacts/
├── fedavg_lr.pkl        # Global Logistic Regression model
├── fedavg_scaler.pkl   # Feature scaler
└── fedavg_meta.json    # Training metadata

```

---

## ⚡ Real-Time Prediction (Critical Design)

### 🔴 No stored predictions are used

When a user asks:

```

What is the risk for patient_id 77?

```

The system:

1. Fetches live data from **all warehouse tables**
2. Dynamically builds feature vector
3. Applies the **federated global model**
4. Returns:
   - Risk label
   - Risk probability
   - Features used

The table `ml_patient_features` exists **only for training and evaluation**, not inference.

---

## 🗃️ Data Warehouse Tables

| Table | Description |
|-----|-------------|
| warehouse_patients | Unified patient demographics |
| warehouse_billing | Billing information |
| warehouse_claims | Insurance claims |
| warehouse_vitals | Patient vitals |
| warehouse_drugs | Drug inventory |
| warehouse_manufacturers | Drug manufacturers |
| warehouse_prescriptions | Prescriptions |
| ml_patient_features | Training feature table |

---

## 🔍 Natural Language Querying

The system supports **hybrid natural-language queries**.

### Analytics Queries (NL → SQL via LLM)
Examples:
```

show patients
show billing
show claims
show vitals
risky patients

```

These are translated to SQL using OpenAI.

---

### Prediction Queries (Hard-routed, No SQL)

Any query containing **predict**, **risk**, or **probability** triggers **real-time inference**:

```

predict risk for patient_id 77
what is the risk for patient 12
risk probability for patient 5

````

No SQL is executed for prediction queries.

---

## 🛠️ Setup Instructions

### 1️⃣ Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

---

### 2️⃣ Configure `.env`

```env
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini

IIA_MODE=db
IIA_RESET_DW=1
IIA_RUN_GENERATORS=1

IIA_MONGO_URI=mongodb://localhost:27017
IIA_MONGO_DB=MedicalRecords
IIA_MONGO_PATIENTS_COLL=Patients

IIA_PG_HOST=localhost
IIA_PG_PORT=5432
IIA_PG_USER=postgres
IIA_PG_PASS=admin
IIA_PG_DB=postgres

IIA_SQLITE_SOURCE_PATH=BillingRecords.db
```

---

## ▶️ Running the System

```bash
python3 backend.py
```

This will:

1. Generate mock data
2. Run ETL
3. Build the data warehouse
4. Train federated models
5. Launch the Flask backend

---

## 🧪 Example Queries

### Analytics

```
show patients
show billing
show claims
show vitals
list prescriptions
risky patients
```

### Prediction

```
predict risk for patient_id 77
what is the risk for patient 12
risk probability for patient 5
```

---

## 🧠 Why This Is Real Federated Learning

✔ Local training per data silo
✔ No raw data sharing
✔ Federated Averaging (FedAvg)
✔ Single global model
✔ Live inference

This is **not ensemble learning**, **not centralized ML**, and **not pseudo-federated**.

---

## 🚀 Future Improvements

* Differential Privacy
* Secure Aggregation
* Client drop-out simulation
* Non-IID data experiments
* Explainable AI (SHAP)
* Streaming vitals ingestion

---

## 👨‍💻 Author

**Kartikeya**

Built as a systems-level project demonstrating:

* Federated Learning
* Multi-database ETL
* Real-time ML inference
* Natural-language analytics


```
