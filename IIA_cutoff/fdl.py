# fdl.py
import Main_run_file

def run_etl_and_train():
    return Main_run_file.run_etl_and_train(reset_dw=True)

def run_etl_periodically(*args, **kwargs):
    return Main_run_file.run_etl_periodically(*args, **kwargs)

def predict_risk(patient_id: int):
    return Main_run_file.predict_risk(patient_id)
