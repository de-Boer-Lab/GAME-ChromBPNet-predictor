import os
import sys
Chrombpnet_SCRIPT_DIR = os.path.dirname(__file__)
Chrombpnet_DIR = os.path.dirname(Chrombpnet_SCRIPT_DIR)
sys.path.append(Chrombpnet_DIR)
from chrombpnet_utils import *

import tensorflow as tf
import chrombpnet.training.utils.one_hot as one_hot

print("TF version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))

def predict_chrombpnet(sequence_dict, request_tasks, matcher_ip, matcher_port, 
                       prediction_ranges=None, is_point_readout=False):
    
    """
    Runs ChromBPNet across all requested tasks and returns full bp-resolution
    track predictions. Point/track readout are applied here as post-processing
    after full predictions are assembled.

    ChromBPNet always predicts at 1bp resolution across the full sequence:
        dim(predictions) = len(sequence)

    prediction_ranges and readout are applied after assembly:
        - track + range: return predictions[start:end+1] of profile head
        - point + range: return mean(predictions[start:end+1]) of profile head
        - track, no range: return full predictions of profile head
        - point, no range: return mean(full predictions) of profile head
                   (profile is already count-weighted: softmax(logits) * exp(logcounts),
                    so mean(profile) = exp(logcounts) / window_size), 
                    where window_size is 1000 for sequences >=1kb and the
                    actual sequence length for sequences <1kb.
    
    Fold averaging:
        - linear scale: arithmetic mean across folds in linear space
        - log scale: geometric mean across folds (average in log space, exp back to linear)
                     predict_across_folds_for_selected_matched_models always returns linear predictions.
                     Final output scale is applied in the server loop via apply_scaling().
        
    Args:
        sequence_dict (dict): {seq_id: sequence_string}
        request_tasks (set of tuples): {(request_type, cell_type, scale_requested), ...}
        matcher_ip (str): IP address of the model matcher service
        matcher_port (int): Port number of the model matcher service
        prediction_ranges (dict, optional): {seq_id: (start, end), ...} specifying prediction ranges. 
                                            Defaults to None.
        is_point_readout (bool, optional): Whether to return point readout (mean over range) instead 
                                           of track readout. Defaults to False.

    Returns:
        tuple: (task_predictions dict, overall_matcher_version str)
               task_predictions is keyed by (request_type, cell_type, scale_actual) 3-tuples.
               scale_actual is applied per-task in the server loop via apply_scaling().
    """
    overall_matcher_version = "N/A"
    task_predictions = {}
    if prediction_ranges is None:
        prediction_ranges = {}

    #Loop through the request tasks
    # print(f"DEBUG: request tasks type: {type(request_tasks)}, value: {request_tasks}")
    for request_type, cell_type, scale_requested in request_tasks:
        
        # Resolve scale -- default to linear if not specified
        if scale_requested is None:
            scale_actual = "linear"
        else:
            scale_actual = scale_requested
        
        # 3-tuple task key: (type, cell_type, scale) so that the same cell type with
        # different scales runs as independent tasks with correct fold averaging each time
        task_key = (request_type, cell_type, scale_actual)

        final_long_seqs_predictions = {}
        final_short_seqs_predictions = {}
        print(f"Making predictions for:{task_key}")
        #Choose the best model for the request
        models_chosen, cell_type_actual, matcher_version = choose_model(cell_type, matcher_ip, matcher_port)
        if matcher_version not in ["N/A", "error"]:
            overall_matcher_version = matcher_version

        if cell_type_actual == None:
            print(f"No matching tracks found for {request_type} and {cell_type}. Skipping...")
            task_predictions[task_key] = {"error": models_chosen}
            continue
       
        task_predictions[task_key] = {
            "cell_type_actual": cell_type_actual,
            "type_actual": models_chosen['Assay'].to_list()
        }

        #Set up the model names for each of the 5 folds
        ENCIDs = models_chosen['ENCID'].to_list()
        model_paths = []
        for encid in ENCIDs:
            for i in range(5):
                model_paths.append(f"model.chrombpnet_nobias.fold_{i}.{encid}.h5")

        #Load all the models once to save time
        model_objects = load_all_models(model_paths)
        sequences = list(sequence_dict.values())
        # Chrombpnet models take 2114bp as input but predict across 1kb
        # since the model is bp resolution we need to return the exact number of predictions 
        # as bp for each seq

        #CASE 1:all the sequences are shorter than 1000bp, you can make 1 prediction
        if all(len(seq) <= 1000 for seq in sequences):
            print("All the sequences are shorter than 1kb")
            sequences_padded = pad_sequences(sequences, 2114)
            one_hot_seqs = one_hot.dna_to_one_hot(sequences_padded)  # shape: (N, 2114, 4)
            # Always pass is_point_readout=False -- full track predictions are always
            # made first. Point averaging and range subsetting are applied below.
            predictions = predict_across_folds_for_selected_matched_models(one_hot_seqs, model_objects, is_point_readout=False,
                                                                           scale_actual=scale_actual)
            
            #Slice the predictions to remove predictions for the padded N bases
            sliced_predictions = slice_predictions(sequences, predictions, 1000)
            for i in range(0,len(sequences)):
                seq_id = list(sequence_dict.keys())[i]
                task_predictions[task_key][seq_id] = sliced_predictions[i]
            
        # CASE 2: Some of the sequences are shorter than 1kb
        else:
            print("Some sequences are be longer than 1kb -- require multiple predictions")
            # Pull the shorter sequences since those can be predicted in one go
            short_seqs_dict = {k: v for k, v in sequence_dict.items() if len(v) <= 1000}
            print(f"There are: {len(short_seqs_dict)} sequences shorter than 1kb")
            # for sequences longer than 1kb we need to make multiple predictions
            long_seqs_dict  = {k: v for k, v in sequence_dict.items() if len(v) > 1000}
            print(f"There are: {len(long_seqs_dict)} sequences longer than 1kb")
            # make all the predictions for the short sequences together, if exists
            if short_seqs_dict:
                sequences_padded = pad_sequences(short_seqs_dict.values(), 2114)
                one_hot_seqs = one_hot.dna_to_one_hot(sequences_padded)  
                # Always pass is_point_readout=False -- full track predictions are always
                # made first. Point averaging and range subsetting are applied below.
                predictions_short_seqs = predict_across_folds_for_selected_matched_models(one_hot_seqs, model_objects, False, scale_actual)
                print("Final Prediction shape for short sequences:", predictions_short_seqs.shape) # (N, 1000)
                #Slice the predictions to remove predictions for the padded bases
                sliced_predictions_short_seqs = slice_predictions(list(short_seqs_dict.values()), predictions_short_seqs, 1000)
            
                # Create a dictionary mapping each key to its prediction
                final_short_seqs_predictions = {key: sliced_predictions_short_seqs[i] 
                        for i, key in enumerate(short_seqs_dict.keys())}

            long_seqs = list(long_seqs_dict.values())
            
            print("Now making predictions for sequence(s) longer than 1kb:", len(long_seqs))
            long_seqs_keys = list(long_seqs_dict.keys())
            
            long_seqs_split = {key: [] for key in long_seqs_keys}
        
            for j in range(0, len(long_seqs)):
                # Mark how much of the actual sequence has been predicted on
                # sequence_length_current = len(long_seqs[j])
                # #print(sequence_length_current)
                #557 is the flanking part of the sequence that is used as input but not predicted on
                #2114 - 1000 = 1114/2
                seq_predicted_end = 557
                start_pos = 0
                sequence_with_upstream_pad = ('N' * (557)) + long_seqs[j]
                
                while seq_predicted_end < len(sequence_with_upstream_pad):
                    #The position is either the start position of the current sequence or the end of the sequence
                    end_pos = min(len(sequence_with_upstream_pad), start_pos+2114)

                    #Current sequence
                    seq_chunk = sequence_with_upstream_pad[start_pos:end_pos]
                    #If there is enough sequence to fit into the model's window don't need to add extra padding
                    #No need to crop the prediction bins either
                    if len(seq_chunk) == 2114:
                        
                        long_seqs_split[long_seqs_keys[j]].append((seq_chunk, 1000))
                        #1kb of the sequence was predicted
                        seq_predicted_end = seq_predicted_end + 1000 
                        start_pos = start_pos + 1000

                    else:
                        #If the sequence doesn't fit into the prediction window, you will need to make 2 predictions to predict for each base
                        if len(seq_chunk) > 1000:
                            #Pad only downstream so sequence is not centered, but the first base is at the start of the first bin
                            downstream_pad = 2114 - len(seq_chunk)
                            
                            seq_chunk_downstreamN = seq_chunk + ('N' * (downstream_pad))
                            #If the seq_chunk with the upstream flank removed is >=1000 we need to keep all the predictions
                            if (len(seq_chunk)-557) >= 1000:
                                to_keep = 1000
                            #otherwise if only need to keep the same number of predictions as the new part of the sequence we predicted on
                            if (len(seq_chunk)-557) < 1000:
                                to_keep = len(seq_chunk)-557
                                
                            long_seqs_split[long_seqs_keys[j]].append((seq_chunk_downstreamN, to_keep))
                            
                            #Next 1kb was predicted
                            #Increment both trackers
                            start_pos = start_pos + 1000
                            seq_predicted_end = seq_predicted_end + 1000

                        #if what's left fits into one prediction window we only need to make one prediction
                        elif len(seq_chunk) <= 1000:
                            #calculate how many Ns we will pad the downstream with
                            downstream_pad = 2114 - len(seq_chunk)

                            seq_chunk_downstreamN = seq_chunk + ('N' * (downstream_pad))
                            long_seqs_split[long_seqs_keys[j]].append((seq_chunk_downstreamN, len(seq_chunk)-557))
                            
                            break
            
            long_seq_only = [seq for v in long_seqs_split.values() for seq, _ in v]
            one_hot_seqs = one_hot.dna_to_one_hot(long_seq_only)  # shape: (N, 2114, 4)
            
            predictions_long_seqs = predict_across_folds_for_selected_matched_models(one_hot_seqs, model_objects, is_point_readout=False, scale_actual=scale_actual)

            final_long_seqs_predictions = slice_predictions_longSeqs(long_seqs_split, predictions_long_seqs, 1000, is_point_readout=False)

        task_predictions[task_key].update({**final_long_seqs_predictions, **final_short_seqs_predictions})
        
        # ----- Apply prediction range slicing and point readout averaging as post-processing after full predictions are assembled -----
        # ChromBPNet always predicts at bp-resolution across the full sequence.
        # prediction_ranges and readout type are applied here as pure post-processing
        # after the full predictions are assembled. 
        for seq_id in sequence_dict.keys():
            raw_pred  = task_predictions[task_key][seq_id]  # np.array, shape (seq_len,)
            seq_range = prediction_ranges.get(seq_id, [])

            if seq_range:
                # Range provided -- crop first, then apply readout
                start, end = seq_range
                cropped = raw_pred[start:end + 1]  # end is inclusive per API spec
                if is_point_readout:
                    task_predictions[task_key][seq_id] = float(np.mean(cropped))
                else:
                    task_predictions[task_key][seq_id] = cropped
            else:
                # Empty range or absent -- full sequence, apply readout only
                if is_point_readout:
                    task_predictions[task_key][seq_id] = float(np.mean(raw_pred))
                # else: full track predictions already in place, no change needed

    return task_predictions, overall_matcher_version

