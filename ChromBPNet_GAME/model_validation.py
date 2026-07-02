import numpy as np
from error_checking_functions import PredictionFailedError

# Define what this specific model supports globally
SUPPORTED_SCALES = ["linear", "log"]
DEFAULT_SCALE = "linear"

def model_specific_payload_validation(payload):
    """
    ChromBPNet-specific validation checks on the payload.
    Runs after general schema validation checks have passed.

    Checks:
        - Readout type: ChromBPNet only supports "point" and "track" readouts, not "interaction_matrix".
        - Species: Must be homo_sapiens for all tasks.
        - Request type: Must be accessibility.
    """
    
    errors = {'prediction_request_failed': []}

    readout_type = payload['readout']

    # Handle unsupported `interaction_matrix` readout
    if readout_type == "interaction_matrix":
        err_msg="ChromBPNet cannot handle 'interaction_matrix' readout type."
        raise PredictionFailedError(f"{err_msg}")

    # --- MODEL SPECIFIC: Ensure this ChromBPNet Predictor only supports homo_sapiens ---
    for task in payload['prediction_tasks']:
        if task.get('species', '').lower() != "homo_sapiens":
            errors['prediction_request_failed'].append(
                f"This predictor only supports species: homo_sapiens."
                f"Received '{task.get('species')}' for task '{task.get('name')}'."
            )
        
    for task in payload['prediction_tasks']:
        if task.get('type', '').lower() != "accessibility":
            errors['prediction_request_failed'].append(
                f"This predictor only supports type: ['accessibility']."
                f"Received '{task.get('type')}' for task '{task.get('name')}'."
            )
            
    # If you want to add error checking that restricts sequences with N bases, add that here
    
    if any(errors.values()):
        flagged_errors = [msg for sublist in errors.values() for msg in sublist]
        raise PredictionFailedError(flagged_errors)

def apply_scaling(predictions_dict, requested_scale):
    """
    Applies scale transformation to assembled predictions.
    Called per-task in the server loop after predictions are returned from
    predict_chrombpnet(). Mirrors the Borzoi apply_scaling pattern.

    ChromBPNet default output: linear (softmax * exp(logcounts)).
    
    Logic: 
      - If 'linear' or None requested: Do nothing, return as is.
      - If 'log' requested: np.log(x), with small epsilon clip to prevent
                            log(0) = -inf. Any residual non-finite values
                            are caught by _sanitize_for_json() in the server.
      
    Args:
        predictions_dict (dict): The raw linear predictions
        requested_scale (str or None): The scale requested by the user
        
    Returns:
        tuple: (transformed_dict, actual_scale_str)
    """
    
    # Determine Effective Scale
    if not requested_scale:
        # Default if None provided
        effective_scale = DEFAULT_SCALE
    else:
        effective_scale = requested_scale.lower()
    
    if effective_scale == "linear":
        return predictions_dict, "linear"
    
    transformed_preds = {}
    for seq_id, values in predictions_dict.items():
        # Convert to numpy for fast vectorized math
        arr = np.array(values, dtype=np.float64)
        
        if effective_scale == "log":
            # Clip to a small positive value before log to prevent -inf from log(0).
            # Any remaining non-finite values (e.g. from upstream NaN) are handled
            # by _sanitize_for_json() in the server after this call.
            arr = np.log(np.clip(arr, 1e-10, None))
        
        # Convert back to list for JSON serialization
        transformed_preds[seq_id] = arr.tolist()
        
    return transformed_preds, effective_scale
    