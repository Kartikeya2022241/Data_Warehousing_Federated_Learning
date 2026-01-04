# import pandas as pd
# from faker import Faker
# from datetime import datetime
# import pymongo
# import json
# '''MongoDB
# db.createCollection("Notes");
# db.createCollection("Images");
# Notes.insertOne({
#     name: "John Doe : 1234567890",
#     diagnosis: "Pneumonia",
#     treatment_plan: "Antibiotics",
#     follow_up: "2 weeks"
# });
# Images.insertOne({
#     name: "John Doe : 1234567890",
#     image: "X-ray"
# });
# '''


# def check_mongodb_collections():
#     try:
#         # Connect to MongoDB
#         client = pymongo.MongoClient("mongodb://localhost:27017/")
#         db = client["hospital"]  # Specify the database

#         # List all collections in the database
#         collections = db.list_collection_names()
#         if not collections:
#             print("No collections found in the database.")
#             return

#         print(f"Collections in the 'hospital' database: {collections}\n")

#         # Iterate through each collection and print documents
#         for collection_name in collections:
#             print(f"Documents in the '{collection_name}' collection:")
#             collection = db[collection_name]

#             # Fetch and display all documents
#             for document in collection.find():
#                 print(document)
#             print("-" * 50)

#     except pymongo.errors.PyMongoError as e:
#         print(f"An error occurred: {e}")
# def main():
#     fake = Faker()
#     patient_data = pd.read_csv("./initial CSV/Patients.csv")
#     patient_notes = []
#     images = []
#     for index,row in patient_data.iterrows():
#         patient_notes.append({
#             "name": row["name"] + " : " + str(row["adhar_number"]),
#             "diagnosis": fake.word(),
#             "treatment_plan": fake.paragraph(),
#             "follow_up": fake.numerify(text="% weeks")
#         })
#         images.append({
#             "name": row["name"] + " : " + str(row["adhar_number"]),
#             "type": fake.word(),
#             "image": fake.uri()
#         })
#     import os
#     if not os.path.exists("./mongodb"):
#         os.makedirs("./mongodb")
#     patient_notes_df = pd.DataFrame(patient_notes)
#     images_df = pd.DataFrame(images)
#     patient_notes_df.to_csv("./mongodb/patient_notes.csv",index=False)
#     images_df.to_csv("./mongodb/images.csv",index=False)
# def load():
#     # load data into mongodb
    
#     client = pymongo.MongoClient("mongodb://localhost:27017/")
#     db = client["hospital"]
#     notes = db["Notes"]
#     images = db["Images"]
#     patient_notes = pd.read_csv("./mongodb/patient_notes.csv")
#     images_csv = pd.read_csv("./mongodb/images.csv")
#     for index,row in patient_notes.iterrows():
#         notes.insert_one(json.loads(row.to_json()))
#     for index,row in images_csv.iterrows():
#         images.insert_one(json.loads(row.to_json()))
#     check_mongodb_collections()
#     print("Data loaded into MongoDB")
# if __name__ == "__main__":
#     check_mongodb_collections()
#     print("Run generate.py file instead")




import pandas as pd
from faker import Faker
from datetime import datetime
import pymongo
import json
import os

def check_mongodb_collections():
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["MedicalRecords"]  # Specify the database

        # List all collections in the database
        collections = db.list_collection_names()
        if not collections:
            print("No collections found in the database.")
            return

        print(f"Collections in the 'hospital' database: {collections}\n")

        # Iterate through each collection and print documents
        for collection_name in collections:
            print(f"Documents in the '{collection_name}' collection:")
            collection = db[collection_name]

            # Fetch and display all documents
            for document in collection.find():
                print(document)
            print("-" * 50)

    except pymongo.errors.PyMongoError as e:
        print(f"An error occurred: {e}")

def main():
    fake = Faker()
    patient_data = pd.read_csv("./initial CSV/Patients.csv")
    patient_notes = []
    images = []

    for index, row in patient_data.iterrows():
        # Split name and adhar_number
        name = row["name"]
        adhar_number = int(row["adhar_number"])

        # Generate fake patient notes
        patient_notes.append({
            "name": name,
            "adhar_number": int(adhar_number),
            "diagnosis": fake.word(),
            "treatment_plan": fake.paragraph(),
            "follow_up": fake.numerify(text="% weeks")
        })

        # Generate fake image data
        # images.append({
        #     "name": name,
        #     "adhar_number": int(adhar_number),
        #     "type": fake.word(),
        #     "image": fake.uri()
        # })

    # Ensure the directory exists
    if not os.path.exists("./mongodb"):
        os.makedirs("./mongodb")

    # Save generated data to CSV files
    patient_notes_df = pd.DataFrame(patient_notes)
    images_df = pd.DataFrame(images)
    patient_notes_df.to_csv("./mongodb/patient_notes.csv", index=False)
    images_df.to_csv("./mongodb/images.csv", index=False)

def load():
    from pymongo import MongoClient
    import random
    from faker import Faker

    # Setup
    fake = Faker()
    client = MongoClient("mongodb://localhost:27017/")
    db = client["MedicalRecords"]

    # Populate Patients
    patients = []
    for i in range(100):
        patients.append({
            "Patient_ID": i + 1,
            "Name": fake.name(),
            "Date_of_Birth": fake.date_of_birth(minimum_age=1, maximum_age=90).isoformat(),
            "Gender": random.choice(["Male", "Female"]),
            "Contact_Details": fake.phone_number(),
            "Emergency_Contact": fake.phone_number(),
        })
    db["Patients"].insert_many(patients)

    # Populate Doctors
    doctors = []
    for i in range(50):  # Assume 50 doctors
        doctors.append({
            "Doctor_ID": i + 1,
            "Name": fake.name(),
            "Specialization": random.choice(["Cardiology", "Neurology", "Oncology", "Pediatrics", "Orthopedics"]),
            "Contact_Details": fake.phone_number(),
        })
    db["Doctors"].insert_many(doctors)

    # Populate Admissions
    admissions = []
    for i in range(100):
        admissions.append({
            "Admission_ID": i + 1,
            "Patient_ID": random.randint(1, 100),
            "Doctor_ID": random.randint(1, 50),
            "Admission_Date": fake.date_this_year().isoformat(),
            "Discharge_Date": fake.date_this_year().isoformat(),
        })
    db["Admissions"].insert_many(admissions)

    # Populate Diagnoses
    diagnoses = []
    for i in range(100):
        diagnoses.append({
            "Diagnosis_ID": i + 1,
            "Admission_ID": random.randint(1, 100),
            "Diagnosis": fake.word() + " syndrome",
            "Diagnosis_Date": fake.date_this_year().isoformat(),
        })
    db["Diagnoses"].insert_many(diagnoses)

    # Populate Drugs
    drugs = []
    for i in range(100):
        drugs.append({
            "Drug_ID": i + 1,
            "Drug_Name": fake.word(),
            "Manufacturer": fake.company(),
            "Expiry_Date": fake.date_this_decade().isoformat(),
            "Batch_Number": fake.uuid4(),
        })
    db["Drugs"].insert_many(drugs)

    # Populate Prescriptions
    prescriptions = []
    for i in range(100):
        prescriptions.append({
            "Prescription_ID": i + 1,
            "Diagnosis_ID": random.randint(1, 100),
            "Drug_ID": random.randint(1, 100),
            "Prescription_Date": fake.date_this_year().isoformat(),
            "Dosage": random.choice(["Once daily", "Twice daily", "Every 8 hours"]),
            "Duration": f"{random.randint(5, 14)} days",
        })
    db["Prescriptions"].insert_many(prescriptions)

    print("MongoDB Medical Records populated.")

    print("Data loaded into MongoDB")

if __name__ == "__main__":
    load()
    check_mongodb_collections()

    # client = pymongo.MongoClient("mongodb://localhost:27017/")
    # db = client["hospital"]
    # print(db.list_collection_names())
    print("Run generate.py file instead")
