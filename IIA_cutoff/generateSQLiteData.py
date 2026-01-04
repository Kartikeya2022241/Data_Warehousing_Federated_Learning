# def load():
#     import pandas as pd
#     # load data into SQLite
#     import sqlite3
#     conn = sqlite3.connect('hospital.db')
#     c = conn.cursor()
#     c.execute('''
#     CREATE TABLE IF NOT EXISTS patients (
#         adhar_number INT PRIMARY KEY,
#         name TEXT,
#         age
#     );
#     ''')
#     c.execute('''
#     CREATE TABLE IF NOT EXISTS billing (
#         invoice_number INT PRIMARY KEY,
#         amount DECIMAL(10, 2),
#         date DATE
#     );
#     ''')
#     c.execute('''
#     CREATE TABLE IF NOT EXISTS patient_billing (
#         patient_id INT,
#         invoice_number INT,
#         PRIMARY KEY (patient_id, invoice_number),
#         FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
#         FOREIGN KEY (invoice_number) REFERENCES billing(invoice_number)
#     );
#     ''')

#     patients = pd.read_csv("./initial CSV/Patients.csv")
#     billing = pd.read_csv("./initial CSV/Billing.csv")
#     patient_billing = pd.read_csv("./initial CSV/Patient_Billing.csv")
#     patients.to_sql('patients', conn, if_exists='replace', index = False)
#     billing.to_sql('billing', conn, if_exists='replace', index = False)
#     patient_billing.to_sql('patient_billing', conn, if_exists='replace', index = False)
#     conn.commit()
#     conn.close()
#     print("Data loaded into SQLite")
# if __name__ == "__main__":
#     print("Run generate.py file instead")



import pandas as pd
import sqlite3
import os
def main():
    """
    Prepare the necessary files and folders for loading data into SQLite.
    This assumes the CSV files are already present in the specified directory.
    """
    # Ensure the directory for input files exists and files are in place
    input_folder = "./initial CSV"
    required_files = ["Patients.csv", "Billing.csv", "Patient_Billing.csv"]

    # Check if all required files are present
    missing_files = [file for file in required_files if not os.path.exists(f"{input_folder}/{file}")]
    if missing_files:
        print(f"Missing files: {', '.join(missing_files)}. Ensure they are in the {input_folder} directory.")
        return
    
    print("All required files are present. Ready for loading into SQLite.")

import sqlite3
import pandas as pd

def load():
    import sqlite3
    import random
    from faker import Faker

    # Setup
    fake = Faker()
    conn = sqlite3.connect("BillingRecords.db")
    cursor = conn.cursor()
    cursor.execute("drop table if exists Billing")
    cursor.execute("drop table if exists Payments")
    cursor.execute("drop table  if exists Insurance_Claims ")

    # Create Tables
    cursor.execute("""
    CREATE TABLE Billing (
        Billing_ID INTEGER PRIMARY KEY,
        Patient_ID INTEGER,
        Admission_ID INTEGER,
        Total_Amount REAL,
        Billing_Date TEXT,
        Payment_Status TEXT
    );
    """)
    cursor.execute("""
    CREATE TABLE Payments (
        Payment_ID INTEGER PRIMARY KEY,
        Billing_ID INTEGER,
        Payment_Method TEXT,
        Amount_Paid REAL,
        Payment_Date TEXT
    );
    """)
    cursor.execute("""
    CREATE TABLE Insurance_Claims (
        Claim_ID INTEGER PRIMARY KEY,
        Billing_ID INTEGER,
        Insurance_Provider TEXT,
        Policy_Number TEXT,
        Claim_Amount REAL,
        Claim_Status TEXT
    );
    """)

    # Populate Billing
    billing_records = []
    for i in range(100):
        billing_records.append((
            i + 1,
            random.randint(1, 100),  # Patient_ID
            random.randint(1, 100),  # Admission_ID
            round(random.uniform(500, 5000), 2),  # Total_Amount
            fake.date_this_year().isoformat(),
            random.choice(["Paid", "Pending", "Overdue"]),
        ))
    cursor.executemany("""
    INSERT INTO Billing (Billing_ID, Patient_ID, Admission_ID, Total_Amount, Billing_Date, Payment_Status)
    VALUES (?, ?, ?, ?, ?, ?);
    """, billing_records)

    # Populate Payments
    payment_records = []
    for i in range(100):
        payment_records.append((
            i + 1,
            random.randint(1, 100),  # Billing_ID
            random.choice(["Credit Card", "Cash", "Insurance"]),
            round(random.uniform(500, 5000), 2),
            fake.date_this_year().isoformat(),
        ))
    cursor.executemany("""
    INSERT INTO Payments (Payment_ID, Billing_ID, Payment_Method, Amount_Paid, Payment_Date)
    VALUES (?, ?, ?, ?, ?);
    """, payment_records)

    # Populate Insurance Claims
    insurance_claims = []
    for i in range(100):
        insurance_claims.append((
            i + 1,
            random.randint(1, 100),  # Billing_ID
            fake.company(),
            fake.uuid4(),
            round(random.uniform(500, 5000), 2),
            random.choice(["Approved", "Pending", "Rejected"]),
        ))
    cursor.executemany("""
    INSERT INTO Insurance_Claims (Claim_ID, Billing_ID, Insurance_Provider, Policy_Number, Claim_Amount, Claim_Status)
    VALUES (?, ?, ?, ?, ?, ?);
    """, insurance_claims)

    conn.commit()
    conn.close()

    print("SQLite Billing and Financial Records populated.")




if __name__ == "__main__":
    load()
    conn = sqlite3.connect('BillingRecords.db')
    c = conn.cursor()

    c.execute("select * from Billing")
    result=c.fetchall()

    for row in result:
        print(row)

    print()


    c.execute("select * from Payments")
    result=c.fetchall()

    for row in result:
        print(row)

    print()

    c.execute("select * from Insurance_Claims")
    result=c.fetchall()

    for row in result:
        print(row)

    print()

    # c.execute("""select b.amount,b.date,p.patient_id,pb.invoice_number,p.adhar_number,p.name,p.age 
    #           from patient_billing as pb join patients 
    #           as p on pb.adhar_number=p.adhar_number join billing as b on b.invoice_number=pb.invoice_number""")
    # result=c.fetchall()
    # print("amount, date, patient_id, invoice_number, adhar_number, name, age")
    # for row in result:
    #     print(row)
    print("Run generate.py file instead")
