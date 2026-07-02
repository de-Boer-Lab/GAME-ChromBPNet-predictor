import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
import chrombpnet.training.utils.losses as losses
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
MATCHER_NULL_RESPONSE = "NULL"


Chrombpnet_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(Chrombpnet_SCRIPT_DIR)
sys.path.append(Chrombpnet_SCRIPT_DIR)

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

def softmax(x, temp=1):
    norm_x = x - np.mean(x,axis=1, keepdims=True)
    return np.exp(temp*norm_x)/np.sum(np.exp(temp*norm_x), axis=1, keepdims=True)

def load_model_wrapper(model_h5):
    # read .h5 model
    custom_objects={"multinomial_nll":losses.multinomial_nll, "tf": tf}    
    get_custom_objects().update(custom_objects)    
    model=load_model(model_h5, compile=False)
    #model.summary()
    return model

def load_all_models(model_path_list):
    model_objects = []
    for model_path in model_path_list:
        full_model_path = f"{Chrombpnet_SCRIPT_DIR}/models/models_nobias/{model_path}"
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(
                f"Model file not found: {full_model_path}. "
                "Ensure all fold .h5 files are present in models/models_nobias/."
            )
        model_objects.append(load_model_wrapper(full_model_path))
    return model_objects

#Need to predict for 5 folds for each models and take the average
def predict_across_folds_for_selected_matched_models(one_hot_encoded_seqs, model_objects, 
                                                     is_point_readout, scale_actual):
    
    """
    Returns linear-scale predictions averaged across all model folds.
    
    Fold averaging depends on the requested scale, consistent with v1 implementation.
        - linear: arithmetic mean across folds in linear space (i.e. average the final predictions)
        - log: geometric mean across folds (average in log space, then exp back to linear)
               This is equivalent to: exp(mean(log(predictions_per_fold)))
    
    NOTE: Always returns linear-scale predictions from this function, even if log scale was requested.
    The scale_actual only controls how fold averaging is done, not the final output scale.
    Final output scaling (e.g. log transformation) is applied downstream via apply_scaling()
    in the server loop.

    NOTE: predict_chrombpnet always calls this with is_point_readout=False so
    that full track predictions are returned. Point readout averaging and
    prediction_ranges subsetting are applied as post-processing in
    predict_chrombpnet after the full track predictions are assembled.
    """
    #load the model
    #Number of sequences x 1000bp x number of models
    num_sequences = one_hot_encoded_seqs.shape[0]
    seq_len = 1000  # assuming predictions are always 1000bp
    num_models = len(model_objects)
    # Temporary array to store all predictions
    if is_point_readout:
        predictions_matrix = np.empty((num_sequences, 1, num_models), dtype=np.float32)
    else:
        predictions_matrix = np.empty((num_sequences, seq_len, num_models), dtype=np.float32)

    if scale_actual == "linear":
        print("making linear scale predictions")
        for idx, model in enumerate(model_objects):
            print(f"predicting on model: {idx}")
            pred_logits_wo_bias, pred_logcts_wo_bias = model.predict(one_hot_encoded_seqs)
            # ^^^^ logits is the profile vector
            # logcts is the single value
            if is_point_readout:
                predictions_matrix[:,:,idx]= np.exp(pred_logcts_wo_bias)
                # print("NEVER REACHED?") # This code path is never reached because predict_chrombpnet always calls with is_point_readout=False, but we keep it here for completeness in case of future use.
            else:
                predictions_matrix[:,:,idx]= softmax(pred_logits_wo_bias) \
                    * (np.expand_dims(np.exp(pred_logcts_wo_bias)[:,0],axis=1)) # final predictions you can use
                # debug print the shape of predictions_matrix after filling
                # print(f"Filled predictions for model {idx}, current shape of predictions_matrix: {predictions_matrix.shape}")
                
        # Arithmetic mean across folds in linear space is just the average of the predictions
        predictions_avg = predictions_matrix.mean(axis=2)
        
    # For log scale   
    elif scale_actual == "log":
        print("making log scale predictions")
        for idx, model in enumerate(model_objects):
            print(f"predicting on model: {idx}")
            pred_logits_wo_bias, pred_logcts_wo_bias = model.predict(one_hot_encoded_seqs)
            if is_point_readout:
                predictions_matrix[:,:,idx]= pred_logcts_wo_bias
                # print("NEVER REACHED?") # This code path is never reached because predict_chrombpnet always calls with is_point_readout=False, but we keep it here for completeness in case of future use.
            else:
                predictions_matrix[:,:,idx]= np.log(
                    softmax(pred_logits_wo_bias) 
                    * (np.expand_dims(np.exp(pred_logcts_wo_bias)[:,0],axis=1))
                ) # final predictions you can use
                # print(f"Filled predictions for model {idx}, current shape of predictions_matrix: {predictions_matrix.shape}")

        # Average across the fold/model axis in log space (geometric mean in linear space), then exp back to linear
        predictions_avg = np.exp(predictions_matrix.mean(axis=2))
    # print("Shape of averaged predictions across models and folds:", predictions_avg.shape)  # (num_sequences, 1000)
    return predictions_avg

def choose_model(cell_type, matcher_ip, matcher_port):
    """
    This function takes in the requested cell type from the Evaluator and tries to find either an exact match or closest matched cell type (using Matcher).
    Args:
        cell_type (str): Requested cell type from Evaluator task
    
    Returns:
        tuple: A tuple containing:
            - Either the rows that map to the cell type that should be used or the request error msg
            - The mapped cell type (cell_type_actual)
            - Matcher versions (if used) otherwise N/A
    """
    request_error_msg = f"Request Error: No match found for: {cell_type}."
    #Read in a .txt file that has the cell type mappings for the models that are included in this container
    model_mappings = pd.read_csv(f"{Chrombpnet_SCRIPT_DIR}/models/model_mappings.txt", header = 0, index_col= None)

    #Check if there is an exact match for the requested cell_type
    # Will extract all rows that have that exact cell line
    model_picked = model_mappings[model_mappings['Cell Line'].str.lower() == cell_type.lower()]

    #if there is no exact match we need to use Matcher
    if model_picked.empty:
        try:
            print(f"No exact matching cell types for: {cell_type}. Querying Matcher for similar cell types")

            matcher_url = f"http://{matcher_ip}:{matcher_port}"
            message_for_Matcher = {
                'cell_type_requested': cell_type,
                'cell_type_list': model_mappings['Cell Line'].unique().tolist()
                }
            try:
                response = requests.post(f"{matcher_url}/match", json=message_for_Matcher) #, timeout=60)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Failed to connect to the remote API at {matcher_url}. Is it running? Error: {e}")
                return "Failed to connect to remote Matcher", None, "error"
                # Parse the JSON response from the server
            matcher_result = response.json()
            print(f"--- Real response from Matcher : {matcher_result} ---")

            matcher_version = matcher_result.get('matcher_version', 'UnknownMatcher')

            # matcher could not find any closely related cell_types
            # NOTE: adding more error checks and using .get(), which will return NoneType if missing, which is seemingly safer for type errors
            if not matcher_result or not matcher_result.get('cell_type_actual') or matcher_result.get('cell_type_actual') == MATCHER_NULL_RESPONSE:
                print("No similar cell types were found using Matcher")
                return request_error_msg, None, matcher_version
            else:
                matched_cell_type = matcher_result['cell_type_actual']
                print(f"Matcher cell type will now be used: {matched_cell_type}")
                
                #Will extract all rows with matched models for the returned cell type
                model_picked = model_mappings[model_mappings['Cell Line'].str.lower() == matched_cell_type.lower()]
                if model_picked.empty:
                    print("Matcher did not find any closely related cell types")
                    return request_error_msg, None, matcher_version
                else:
                    return model_picked, matched_cell_type, matcher_version
        except ConnectionError as e:
            print(f"A fatal error occurred while communicating with the Matcher: {e}")
            error_message = f"Internal Server Error: The dependent Matcher service at {matcher_ip}:{matcher_port} is unavailable."
            # Return a 3-element tuple to match the success signature and avoid crashing the caller
            return error_message, None, "error"
    else:

        return model_picked, cell_type, "N/A"

def pad_sequences(sequences, target_length):
    """
    Pad a sequence with 'N' until it reached the target length.

    Args:
        seq (str): The input sequence.
        target_length (int): Model-dependent desired length of the sequence

    Returns:
        padded_seq: The padded sequence.
    """
    padded_list = []
    for seq in sequences:
        seq_len = len(seq)

        # If sequence length is less than target_length, excluding adapters, simply
        # pad with Ns until it is target_length and then add the adapters.
        if seq_len < target_length:
            total_padding = target_length - seq_len
            right_padding = 'N' * (total_padding // 2)
            left_padding = 'N' * (total_padding - len(right_padding))
            padded_seq = left_padding + seq + right_padding
            padded_list.append(padded_seq)
        # When the sequence length, excluding adapters, is the target_length,
        # simply add adapters to each side.
        elif seq_len == target_length:
            padded_list.append(seq)
        else:
            # FIX: sequences longer than target_length were previously silently
            # dropped from padded_list, causing an index mismatch when predictions
            # were zipped back to sequence IDs downstream.
            raise ValueError(
                f"pad_sequences: sequence of length {seq_len} exceeds target "
                f"{target_length}. Sequences longer than {target_length}bp must "
                "be handled by the long-sequence chunking path before padding."
            )

    return padded_list

def slice_predictions(sequences, predictions, target_length):
    sliced_predictions = []
    
    for i in range(0,len(sequences)):
        seq_len = len(sequences[i])

        if seq_len < target_length:
            total_padding = target_length - seq_len
            right_padding = 'N' * (total_padding // 2)
            left_padding = 'N' * (total_padding - len(right_padding))
            start = len(left_padding)
            sliced_predictions.append(predictions[i,start:(start+seq_len)])

        #If the sequence length is the target length no need to slice
        elif seq_len == target_length:
            sliced_predictions.append(predictions[i,:])
    return sliced_predictions

def slice_predictions_longSeqs(sequence_dict, predictions, target_length, is_point_readout):
    sliced_predictions = []

    if is_point_readout == False:
        
        i = 0
        for key, value_list in sequence_dict.items():
            for seq, length in value_list:
                #print(f"Sequence length: {length}")
                if length < 1000:
                    #Padding was only added downstream so just need to take the length of sequence from predictions
                    #print(len(predictions[i,0:length]))
                    sliced_predictions.append(predictions[i,0:length])
                elif length == target_length:
                    #print("No need to slice predictions")
                    sliced_predictions.append(predictions[i,:])
                i+=1
    else:
        #No need to slices the point predictions, only get 1 value per sequence
        sliced_predictions = predictions
    #print(sequence_dict)
    seq_to_squished_predictions = {}
    pred_index = 0
    for seq_key, seq_list in sequence_dict.items():
        squished_per_sequence = []
        for chunk, length in seq_list:
            
            # append prediction for this chunk
            squished_per_sequence.append(sliced_predictions[pred_index])
            pred_index += 1
            # squish predictions for this sequence

        if is_point_readout:
            #Here you are taking the mean of the point count values
            seq_to_squished_predictions[seq_key] = [np.mean(np.concatenate(squished_per_sequence, axis=0))]
        else:
            #print("FINAL length")
            #print(len(np.concatenate(squished_per_sequence, axis=0)))
            seq_to_squished_predictions[seq_key] = np.concatenate(squished_per_sequence, axis=0)

    return seq_to_squished_predictions