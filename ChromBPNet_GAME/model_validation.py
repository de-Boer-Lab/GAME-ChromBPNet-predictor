import socket
import json
import numpy as np
import random
import base64

from error_checking_functions import *

def model_specific_payload_validation(payload):
    
    errors = {'prediction_request_failed': []}

    readout_type = payload['readout']
    is_point_readout = readout_type == "point"

    # Handle unsupported `interaction_matrix` readout
    if readout_type == "interaction_matrix":
        print("ChromBPNet cannot handle 'interaction_matrix' readout type. Exiting gracefully!")
        errors['prediction_request_failed'].append("ChromBPNet cannot process 'interaction_matrix' readout type.")

    # --- MODEL SPECIFIC: Ensure this Enformer Predictor only supports homo_sapiens ---
    for task in payload['prediction_tasks']:
        if task.get('species', '').lower() != "homo_sapiens":
            errors['prediction_request_failed'].append(
                f"This predictor only supports species: homo_sapiens. Received '{task.get('species')}' for task '{task.get('name')}'."
            )
        
    for task in payload['prediction_tasks']:
        if task.get('type', '').lower() != "accessibility":
            errors['prediction_request_failed'].append(
                f"This predictor only supports type: ['accessibility']. Received '{task.get('type')}' for task '{task.get('name')}'."
            )
            
    #If you want to add error checking that restricts sequences with N bases, add that here
    if any(errors.values()):
        flagged_errors = [msg for sublist in errors.values() for msg in sublist]
        raise PredictionFailedError(flagged_errors)
