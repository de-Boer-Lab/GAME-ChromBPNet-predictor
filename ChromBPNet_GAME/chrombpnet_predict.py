import json
import pandas as pd
import requests
import os
import sys
Chrombpnet_SCRIPT_DIR = os.path.dirname(__file__)
Chrombpnet_DIR = os.path.dirname(Chrombpnet_SCRIPT_DIR)
sys.path.append(Chrombpnet_DIR)
from chrombpnet_utils import *


import tensorflow as tf
from tensorflow.keras.models import load_model
import chrombpnet.training.utils.losses as losses
import chrombpnet.training.utils.one_hot as one_hot
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

print("TF version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))

def predict_chrombpnet(sequence_dict, request_tasks, matcher_ip, matcher_port, is_point_readout=False):
    overall_matcher_version = "N/A"
    task_predictions = {}

    #Loop through the request tasks
    for request_type, cell_type, scale_requested in request_tasks:
        task_key = (request_type, cell_type)

        #If no specific scale is request then defalt to linear
        if scale_requested is None:
            scale_actual = "linear"
        else:
            scale_actual = scale_requested

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
        else:
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
            #Chrombpnet model take 2114bp as input but predict across 1kb
            #since the model is bp resolution we need to return the exact number of predictions as bp for each seq

            #CASE 1:all the sequences are shorter than 1000bp, you can make 1 prediction
            if all(len(seq) <= 1000 for seq in sequences):
                print("All the sequences are shorter than 1kb")
                sequences_padded = pad_sequences(sequences, 2114)
                one_hot_seqs = one_hot.dna_to_one_hot(sequences_padded)  # shape: (N, 2114, 4)
                predictions = predict_across_folds_for_selected_matched_models(one_hot_seqs, model_objects, is_point_readout, scale_actual)
                print(predictions)
                
                #If point readout was requested the dimensions are 1000x1
                if is_point_readout:
                    for i in range(0,len(sequences)):
                        seq_id = list(sequence_dict.keys())[i]
                        task_predictions[task_key][seq_id] = predictions[i]
                else:
                    #Slice the predictions to remove predictions for the padded N bases for track predictions
                    sliced_predictions = slice_predictions(sequences, predictions, 1000)
                    for i in range(0,len(sequences)):
                        seq_id = list(sequence_dict.keys())[i]
                        task_predictions[task_key][seq_id] = sliced_predictions[i]
            #CASE 2: Some of the sequences are shorter than 1kb
            else:
                print("Some sequences are be longer than 1kb - require multiple predictions")
                #Pull the shorter sequences since those can be predicted in one go
                short_seqs_dict = {k: v for k, v in sequence_dict.items() if len(v) <= 1000}
                print(f"There are: {len(short_seqs_dict)} sequences shorter than 1kb")
                #for sequences longer than 1kb we need to make multiple predictions
                long_seqs_dict  = {k: v for k, v in sequence_dict.items() if len(v) > 1000}
                print(f"There are: {len(long_seqs_dict)} sequences longer than 1kb")
                #make all the predictions for the short sequences together
                sequences_padded = pad_sequences(short_seqs_dict.values(), 2114)
                one_hot_seqs = one_hot.dna_to_one_hot(sequences_padded)  

                predictions_short_seqs = predict_across_folds_for_selected_matched_models(one_hot_seqs, model_objects, is_point_readout, scale_actual)
                print("Final Prediction shape for short sequences:", predictions_short_seqs.shape) # (N, 1000)

                if is_point_readout:
                    # Create a dictionary mapping each key to its prediction
                    final_short_seqs_predictions = {key: predictions_short_seqs[i] 
                            for i, key in enumerate(short_seqs_dict.keys())}
                else:
                    #Slice the predictions to remove predictions for the padded bases
                    sliced_predictions_short_seqs = slice_predictions(list(short_seqs_dict.values()), predictions_short_seqs, 1000)
               
                    # Create a dictionary mapping each key to its prediction
                    final_short_seqs_predictions = {key: sliced_predictions_short_seqs[i] 
                            for i, key in enumerate(short_seqs_dict.keys())}

                long_seqs = list(long_seqs_dict.values())
               
                print("Now making predictions for the sequences longer than 1kb:", len(long_seqs))
                long_seqs_keys = list(long_seqs_dict.keys())
              
                long_seqs_split = {key: [] for key in long_seqs_keys}
            
                for j in range(0, len(long_seqs)):
                    #Mark how much of the actual sequence has been predicted on
                    sequence_length_current = len(long_seqs[j])
                    print(sequence_length_current)
                    #557 is the flanking part of the sequence that is used as input but not predicted on
                    #2114 - 1000 = 1114/2
                    seq_predicted_end = 557
                    start_pos = 0
                    sequence_with_upstream_pad = ('N' * (557)) + long_seqs[j]
                   
                    while seq_predicted_end < len(sequence_with_upstream_pad):
                        #The position is either the start position of the current sequence or the end of the sequence
                        end_pos = min(len(sequence_with_upstream_pad), start_pos+2114)
                        print("HERE")
                        print(end_pos)
                        print(start_pos)
                        #Current sequence
                        seq_chunk = sequence_with_upstream_pad[start_pos:end_pos]
                        print(len(seq_chunk))
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
                                #If the seq_chunck with the upstream flank removed is >=1000 we need to keep all the predictions
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
                
                predictions_long_seqs = predict_across_folds_for_selected_matched_models(one_hot_seqs, model_objects, is_point_readout, scale_actual)

                final_long_seqs_predictions = slice_predictions_longSeqs(long_seqs_split, predictions_long_seqs, 1000, is_point_readout)

            task_predictions[task_key].update({**final_long_seqs_predictions, **final_short_seqs_predictions})

    return task_predictions, overall_matcher_version, scale_actual

