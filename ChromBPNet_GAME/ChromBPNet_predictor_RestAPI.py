'''RESTful Test Evaluator Utilizing Flask'''
import json
import math
import argparse
from flask import Flask

from config import PREDICTOR_NAME, HELP_FILE, SUPPORTED_REQUEST_FORMATS, SUPPORTED_RESPONSE_FORMATS
from error_checking_functions import *
from schema_validation import *
from chrombpnet_predict import *
from predictor_content_handler import decode_request, encode_response
from model_validation import *
from chrombpnet_utils import *

# --- Have arguments be defined globally ---
parser = argparse.ArgumentParser(description=f'{PREDICTOR_NAME} Predictor API')
parser.add_argument('ip', type=str, help='IP address to bind')
parser.add_argument('port', type=int, help='Port to bind')
parser.add_argument('matcher_ip', type=str, help='Matcher Service IP')
parser.add_argument('matcher_port', type=int, help='Matcher Service Port')

args = parser.parse_args()

predictor_ip = args.ip
predictor_port = args.port
matcher_ip = args.matcher_ip
matcher_port = args.matcher_port

print(f"Matcher service configured at: {matcher_ip}:{matcher_port}")

# ----------- Sanitization -----------
# Clips non-finite floats (NaN, Inf) before JSON serialization.
# Important for log-scale predictions where log(0) -> -inf.
MAX_VALUE = 1e5
MIN_VALUE = -1e5

def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return MAX_VALUE if obj > 0 else MIN_VALUE
        return obj
    return obj

# --- Flask App and Central Error Handler ---
app = Flask(__name__)
# One of these works to maintain order when using jsonify()
app.config["JSON_SORT_KEYS"] = False
app.json.sort_keys = False

def create_error_response(error_key, messages, status_code):
    """ 
    Formats error response into a standardized JSON structure.
    
    Args:
        error_key (str): The category of the error (e.g. 'bad_prediction_request', 'prediction_request_failed').
        messages (list or str): A list of error message strings or a single message.
        status_code (int): Standard HTTP error status code based on the error.
    
    Returns:
        dict: A dictionary formatted for the standardized JSON error response.
    """
    if not isinstance(messages, list):
        messages = [str(messages)]
    error_payload = {"error": [{error_key: msg} for msg in messages]}
    print(error_payload)
    return error_payload, status_code

@app.errorhandler(APIError)
def handle_api_error(error):
    """This single handler catches all of our custom API errors."""
    # Get raw payload and status code
    payload, status_code = create_error_response(error.error_key, error.message, error.status_code)
    
    return encode_response(
        payload, 
        status_code=status_code,
        isError=True,
        predictor_name=PREDICTOR_NAME)
    

@app.after_request
def after_request_callback(response):
    """This function runs after each request is processed."""
    print(f"\n--- Sending predictions back to Evaluator. ---")
    print(f"--- Request Complete. {PREDICTOR_NAME} Predictor is listening on http://{predictor_ip}:{predictor_port} ---\n")
    return response

# --- API Endpoints ---
@app.route('/formats', methods=['GET'])
def formats_endpoint():
    """Provides the Predictor's supported formats"""
    supported_fmts = {
        "predictor_supported_request_formats": SUPPORTED_REQUEST_FORMATS,
        "predictor_supported_response_formats": SUPPORTED_RESPONSE_FORMATS
    }
    try:
        return encode_response(
            supported_fmts,
            status_code=200,
            predictor_name=PREDICTOR_NAME,
            supported_response_formats=SUPPORTED_RESPONSE_FORMATS)
    except Exception as e:
        raise ServerError(f"Error serializing supported format for /format endpoint: {e}")

@app.route('/help', methods=['GET'])
def help_endpoint():
    """Provides the Predictor's help/metadata information."""
    try:
        with open(HELP_FILE, 'r') as f:
            help_data = json.load(f)
        return encode_response(
            help_data,
            status_code=200,
            predictor_name=PREDICTOR_NAME,
            supported_response_formats=SUPPORTED_RESPONSE_FORMATS)
    except Exception as e:
        raise ServerError(f"Error reading help file: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    """The main endpoint for receiving sequences and returning predictions."""
    
    try:
        #Decode incoming request, using the headers or JSON default
        evaluator_request = decode_request(SUPPORTED_REQUEST_FORMATS)
            
        # Validate the payload using the imported function
        # These functions will raise an APIError on failure,
        # which will be caught automatically by @app.errorhandler
        validate_request_payload(evaluator_request)
        readout_type = evaluator_request['readout']
        is_point_readout = readout_type == "point"
        #Model specific error checking should go here
        model_specific_payload_validation(evaluator_request)

        # Preprocess the data using the imported function
        sequences = preprocess_data(evaluator_request)
       
        prediction_ranges = evaluator_request.get('prediction_ranges', {})
        
        # ---------------------- Extract Prediction Tasks and Run the Model ----------------------
        # Start big loop here for all the prediction_tasks
        # First step is to collect all unique tasks
        request_tasks = set()  # Store unique (request_type, cell_type, scale_requested) tuples
        for prediction_task in evaluator_request['prediction_tasks']:
            request_type = prediction_task['type']
            cell_type = prediction_task['cell_type']
            scale_requested = prediction_task.get("scale", None)
            request_tasks.add((request_type, cell_type, scale_requested))

        print(f"Unique tasks extracted: {request_tasks}")
        # Then run ChromBPNet Model ONCE for all required tracks
        print("Running ChromBPNet model on collected tasks...")
        task_predictions, matcher_version = predict_chrombpnet(sequences, request_tasks, matcher_ip,
                                                               matcher_port, prediction_ranges,
                                                               is_point_readout)
        
        model_errors = {'prediction_request_failed': []}
        if isinstance(task_predictions, str):
            # Wrap the error string into error payload 
            model_errors[
                'prediction_request_failed'].append(task_predictions)
            print("Model error; sending error JSON")

        if any(model_errors.values()):
            flagged_errors = [msg for sublist in model_errors.values() for msg in sublist]
            raise PredictionFailedError(flagged_errors)

        task_predictions = convert_numpy_types(task_predictions)
        # Now format predictions to API JSON structure
        # Create JSON to return
        json_return = {
            'matcher_version': matcher_version,
            'bin_size': 1,
            # Prediction task is an array of objects for all requested tasks
            'prediction_tasks': []
        }

        # Loop through all the prediction tasks
        for prediction_task in evaluator_request['prediction_tasks']:
            request_type = prediction_task['type']
            cell_type = prediction_task['cell_type']
            requested_scale = prediction_task.get('scale', None)

            # 3-tuple lookup must match the key used in predict_chrombpnet
            # scale_actual resolves None -> "linear" inside predict_chrombpnet,
            # so we resolve the same way here for the lookup
            scale_actual_for_lookup = requested_scale if requested_scale is not None else "linear"
            task_key = (request_type, cell_type, scale_actual_for_lookup)
            task_result = task_predictions[task_key]
           
            predictions = {
                seq_id: result
                for seq_id, result in task_result.items()
                if seq_id not in ['cell_type_actual', 'type_actual']
            }

            if "error" in predictions:
                # Create structured response for the evaluator
                current_prediction_task = {
                    'name': prediction_task['name'],
                    'type_requested': request_type,
                    'type_actual': "N/A",
                    'cell_type_requested': cell_type,
                    'cell_type_actual': "N/A",
                    'species_requested': prediction_task['species'],
                    'species_actual': prediction_task['species'],
                    'scale_prediction_requested': requested_scale, 
                    'scale_prediction_actual': "N/A",
                    'predictions': predictions
                    
                }
            else:
                
                # NOTE: sanitize before scaling -- clamps any NaN/Inf to finite values.
                # A second pass may be needed if log scale produces -inf from log(0),
                # which is handled by the epsilon clip in apply_scaling().
                predictions = _sanitize_for_json(predictions)
                
                predictions_scaled, effective_scale = apply_scaling(predictions, requested_scale)

                current_prediction_task = {
                    'name': prediction_task['name'],
                    'type_requested': request_type,
                    'type_actual': task_result['type_actual'],
                    'cell_type_requested': cell_type,
                    'cell_type_actual': task_result['cell_type_actual'],
                    'species_requested': prediction_task['species'],
                    'species_actual': prediction_task['species'],
                    'scale_prediction_requested': requested_scale,
                    'scale_prediction_actual': effective_scale,
                }

                # Only add aggregation if there are multiple types
                if len(task_result['type_actual']) > 1:
                    current_prediction_task['aggregation'] = {"models": "mean"}

                current_prediction_task['predictions'] = predictions_scaled
            
            # Append results for current prediction task to the main JSON object
            json_return['prediction_tasks'].append(current_prediction_task)
        
        final_payload = {"predictor_name": PREDICTOR_NAME,
                         **json_return}
        return encode_response(
            final_payload,
            status_code=200,
            predictor_name=PREDICTOR_NAME,
            supported_response_formats=SUPPORTED_RESPONSE_FORMATS)
    
    
    except Exception as e:
        # If it's already an APIError, re-raise it for the handler
        if isinstance(e, APIError):
            raise e
        # Otherwise, wrap the unknown error in a ServerError
        raise ServerError(f"An unexpected internal error occurred: {e}.")


if __name__ == "__main__":
    print(f"{PREDICTOR_NAME} Predictor is starting up/ running  on http://{predictor_ip}:{predictor_port}")
    app.run(host=predictor_ip, port=predictor_port)
