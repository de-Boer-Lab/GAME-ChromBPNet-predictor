'''RESTful Test Evaluator Utilizing Flask'''
import os
import sys
import json
from flask import Flask

from error_checking_functions import *
from schema_validation import *
from chrombpnet_predict import *
from predictor_content_handler import decode_request, encode_response
from model_validation import *
from chrombpnet_utils import *
# Get the absolute path of the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Hardcode name of this Predictor. It will be added to ALL responses.
PREDICTOR_NAME = "ChromBPNet"

# Determine if running inside a container or not
if os.path.exists('/.singularity.d'):
    # Running inside the container
    print("Running inside the container...🥡")
    HELP_FILE = "/predictor_help_message.json"
else:
    # Running outside the container
    print("Running outside the container...📋")
    PREDICTOR_CONTAINER_DIR = os.path.dirname(SCRIPT_DIR)
    HELP_FILE = os.path.join(SCRIPT_DIR, 'predictor_help_message.json')


# ------ Configuration for Wire-Format ------
SUPPORTED_REQUEST_FORMATS = [fmt.lower() for fmt in ["application/json", "application/msgpack"]]
SUPPORTED_RESPONSE_FORMATS = [fmt.lower() for fmt in ["application/json", "application/msgpack"]] # JSON is always supported even when not mentioned. This is jsut to show that. 

# --- Flask App and Central Error Handler ---
app = Flask(__name__)
# One of these works to maintain order when using jsonify()
app.config["JSON_SORT_KEYS"] = False
app.json.sort_keys = False

def create_error_response(error_key, messages, status_code):
    """ 
    Formats error response into a standarized JSON structure.
    
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
        readout_type = evaluator_request.get('readout')
        
        # ---------------------- Extract Prediction Tasks and Run the Model ----------------------
        # Start big loop here for all the prediction_tasks
        # First step is to collect all unique tasks
        request_tasks = set()  # Store unique (request_type, cell_type) pairs
        for prediction_task in evaluator_request['prediction_tasks']:
            request_type = prediction_task['type']
            cell_type = prediction_task['cell_type']
            scale_requested = prediction_task.get("scale", None)

            request_tasks.add((request_type, cell_type, scale_requested))

        print(f"Unique tasks extracted: {request_tasks}")
        # Then run ChromBPNet Model ONCE for all required tracks
        print("Running ChromBPNet model on collected tasks...")
        task_predictions, matcher_version, scale_actual = predict_chrombpnet(sequences, request_tasks, matcher_ip, matcher_port, is_point_readout)
        
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
            task_name = prediction_task['name']
            request_type = prediction_task['type']
            cell_type = prediction_task['cell_type']

            # Retrieve the predictions for this task
            task_key = (request_type, cell_type)
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
                    'type_actual': "N/A",  # If remapped, update this
                    'cell_type_requested': cell_type,
                    'cell_type_actual': "N/A",  # If remapped, update this
                    'species_requested': prediction_task['species'],
                    'species_actual': prediction_task['species'],
                    'scale_prediction_requested': prediction_task.get('scale', None), 
                    'scale_prediction_actual': "N/A",
                    'predictions': predictions
                    
                }
            else:

                current_prediction_task = {
                    'name': prediction_task['name'],
                    'type_requested': request_type,
                    'type_actual': task_result['type_actual'],  # If remapped, update this
                    'cell_type_requested': cell_type,
                    'cell_type_actual': task_result['cell_type_actual'],  # If remapped, update this
                    'species_requested': prediction_task['species'],
                    'species_actual': prediction_task['species'],
                    'scale_prediction_requested': prediction_task.get('scale', None),
                    'scale_prediction_actual': scale_actual,
                }

                # Only add aggregation if there are multiple types
                if len(task_result['type_actual']) > 1:
                    current_prediction_task['aggregation'] = {"models": "mean"}

                current_prediction_task['predictions'] = predictions
            
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
    # if len(sys.argv) != 5:
    #     print(f"Invalid arguments! Arguments must have: <container image/python script> <ip_address> <port> <matcher_ip_address> <matcher_port>")
    #     sys.exit(1)
        
    predictor_ip = sys.argv[1]
    predictor_port = int(sys.argv[2])
    matcher_ip = sys.argv[3]
    matcher_port = sys.argv[4]
    app.run(host=predictor_ip, port=predictor_port)
