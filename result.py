from pathlib import Path
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import os
import yaml
import pprint

import scienceplots
plt.style.use('science')

# ---------- GENERAL

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        return data
    
def print_config_yaml(base_path):
    config_path = Path(base_path) / ".hydra" / "config.yaml"
    config = read_yaml(config_path)
    pprint.pprint(config)
    return config
    
def print_pickle(path):
    try:
        with open(path, 'rb') as file:
            # Load the content of the pickle file
            content = pickle.load(file)
            # Check if the content is a list
            if isinstance(content, list):
                # Convert all elements to strings (if they aren't already)
                content_str = [str(item) for item in content]
                # Join the string representations of the elements and print
                print(', '.join(content_str))
                return content_str
            else:
                # If content is not a list, print it directly
                print(content)
    except FileNotFoundError:
        print(f"The file at {path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
# import os
# import tensorflow as tf
# def read_tfevents_file(base_path, metrics):
#     """
#     Find the first tfevents file within the specified base path and collect values for specified metrics.
    
#     Parameters:
#     - base_path: The base directory to search in for the tfevents file.
#     - metrics: Either a single metric name as a string, or a list of metric names to collect.
    
#     Returns:
#     - If metrics is a single string, returns a list of values found for that metric.
#     - If metrics is a list of strings, returns a dictionary where each key is a metric and the value is a list of values found for that metric.
#     - Returns None if no tfevents file is found.
#     """
#     # Determine if a single metric or a list of metrics is provided
#     single_metric = isinstance(metrics, str)
#     if single_metric:
#         metrics = [metrics]  # Convert to list for uniform processing
    
#     # Find the tfevents file
#     tfevents_file_path = None
#     for root, dirs, files in os.walk(base_path):
#         for file in files:
#             if "tfevents" in file:
#                 tfevents_file_path = os.path.join(root, file)
#                 break
#         if tfevents_file_path:
#             break

#     if not tfevents_file_path:
#         print("No tfevents file found in the specified base path.")
#         return None

#     # Initialize storage for metric values
#     metrics_values = {metric: [] for metric in metrics}
    
#     # Read and collect metrics from the tfevents file
#     for e in tf.compat.v1.train.summary_iterator(tfevents_file_path):
#         for v in e.summary.value:
#             if v.tag in metrics and v.HasField('simple_value'):
#                 metrics_values[v.tag].append(v.simple_value)
    
#     # Return the appropriate format based on input
#     if single_metric:
#         return metrics_values[metrics[0]]  # Return list directly if only one metric was requested
#     else:
#         return metrics_values  # Return dictionary of lists if multiple metrics were requested

# ---------- TABLE RESULT

def weighted_average(group):
    # Calculate the sum of total_examples for the current group
    total_examples_sum = group['total_examples'].sum()
    
    # Initialize the dictionary with keys that will not undergo weighted average calculation
    weighted_avg = {
        'total_examples': total_examples_sum,
        'dataset': 'Overall',  # Set the dataset name
        'round': group['round'].iloc[0]  # The round number is the same for all entries in the group
    }
    
    # Iterate over columns to calculate weighted average for applicable columns
    for column in group.columns:
        if column not in ['total_examples', 'dataset', 'round']:
            weighted_avg[column] = (group[column] * group['total_examples']).sum() / total_examples_sum
    
    return pd.Series(weighted_avg)

# single-indexed df with columns: dataset, round, type, accuracy, subset, metric
def history_to_df_melted(base_path, history_key, metric_type):
    # Read history from pickle file
    with open(str(Path(base_path) / "results.pkl"), 'rb') as file:
        history = pickle.load(file)['history']
        history = history.__dict__[history_key] # metrics_distributed or metrics_distributed_fit

    # Convert dictionary to DataFrame
    results = []
    for key, value in history.items():
        dataset = key
        for rnd, metrics in value:
            metrics['dataset'] = dataset.upper()
            metrics['round'] = rnd
            results.append(metrics)
    df = pd.DataFrame(results)
    
    try:
        # Calculate weighted averages
        weighted_averages_df = df.groupby('round', group_keys=False).apply(weighted_average).reset_index(drop=True)
        df = pd.concat([df, weighted_averages_df])
    except KeyError:
        return None
        
    # Selecting columns based on user input
    value_vars = df.columns[[metric_type in col for col in df.columns]]

    # Melting the DataFrame
    df_melted = df.melt(id_vars=['dataset', 'round'], value_vars=value_vars, 
                        var_name='type', value_name=metric_type)
    
    # Split the column into three new columns
    df_melted[['subset', 'metric', 'type']] = df_melted['type'].str.split('_', expand=True)
    
    return df_melted

# multi-indexed df with information: train, val, test accuracy for client and server for each dataset
def history_to_df_summary(base_path, metric_type, rnd='last'):
    df_melted_fit = history_to_df_melted(base_path, 'metrics_distributed_fit', metric_type)
    
    if df_melted_fit is None:
        return None
    
    if rnd == 'last':
        rnd = df_melted_fit['round'].max()
        print("Last Round:", rnd)
    
    df_melted_fit = df_melted_fit[df_melted_fit['round'] == rnd]
    df_fit = df_melted_fit.groupby(['dataset', 'subset', 'type']).last().unstack(level=['subset', 'type'])[metric_type]

    df_melted_test = history_to_df_melted(base_path, 'metrics_distributed', metric_type)
    df_melted_test = df_melted_test[df_melted_test['round'] == rnd]
    df_test = df_melted_test.groupby(['dataset', 'subset', 'type']).last().unstack(level=['subset', 'type'])[metric_type]

    df_concat = pd.concat([df_fit, df_test], axis=1)

    df_concat.index = pd.CategoricalIndex(df_concat.index, categories=['DEAP', 'SEED', 'DREAMER', 'Overall'], ordered=True)
    df_concat.sort_index(inplace=True)
    
    return df_concat

def read_multirun(multirun_path, metric_type, rnd='last'):
    all_summary = []
    for run_dir in sorted(os.listdir(multirun_path)):
        
        config_path = os.path.join(multirun_path, run_dir, '.hydra', 'overrides.yaml')
        
        if os.path.exists(config_path):
            config_data = read_yaml(config_path)
            config_dict = {item.split('=')[0]: float(item.split('=')[1]) if item.split('=')[1].replace('.', '', 1).isdigit() else item.split('=')[1] for item in config_data}

            base_path = os.path.join(multirun_path, run_dir)
            results_path = os.path.join(base_path, 'results.pkl')
            if os.path.exists(results_path):
                df = history_to_df_summary(base_path, metric_type, rnd)
                
                if df is None:
                    continue
                
                for k, v in config_dict.items():
                    df[k] = v
                
                config_dict_keys = list(config_dict.keys())
                df.set_index(config_dict_keys, append=True, inplace=True)
                df = df.reorder_levels(config_dict_keys + ['dataset'])
                
                all_summary.append(df)
                
    return pd.concat(all_summary).sort_index()

# ---------- PLOT RESULT

def viz_single_train_val(base_path, metric_type):
    # Data preparation
    df_melted = history_to_df_melted(base_path, 'metrics_distributed_fit', metric_type)

    # Visualization
    hue_order = ['DEAP', 'SEED', 'DREAMER', 'Overall']
    hue_color = ['red', 'green', 'blue', 'black']

    for fit_type in ['server', 'client']:
        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(data=df_melted[df_melted['type'] == fit_type], 
                        x='round', y=metric_type, 
                        hue='dataset', hue_order=hue_order, palette=hue_color, 
                        style='subset', markers='o')

        # Customizing line styles immediately after plotting
        lines = ax.get_lines()
        for line in lines:
            label = line.get_label()
            if label in ['_child0', '_child2', '_child4']:
                line.set_alpha(0.4)  # Make these lines a bit transparent
            elif label == '_child6':
                line.set_linewidth(2)  # Make the line bolder

        plt.title(f'{fit_type.capitalize()} Model Train and Validation {metric_type.capitalize()}')
        plt.xlabel('Round')
        plt.ylabel(metric_type.capitalize())
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=8)
        if metric_type == 'accuracy':
            plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        
        plt.tight_layout()
        plt.show()

def viz_single_test(base_path, metric_type):
    # Data preparation
    df_melted = history_to_df_melted(base_path, 'metrics_distributed', metric_type)

    # Visualization
    hue_order = ['DEAP', 'SEED', 'DREAMER', 'Overall']
    hue_color = ['red', 'green', 'blue', 'black']

    for model_type in ['server', 'client']:
        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(data=df_melted[df_melted['type'] == model_type], 
                        x='round', y=metric_type, 
                        hue='dataset', hue_order=hue_order, palette=hue_color, 
                        markers='o')

        # Customizing line styles immediately after plotting
        lines = ax.get_lines()
        for line in lines:
            label = line.get_label()
            if label in ['_child0', '_child2', '_child4']:
                line.set_alpha(0.4)  # Make these lines a bit transparent
            elif label == '_child6':
                line.set_linewidth(2)  # Make the line bolder

        plt.title(f'{model_type.capitalize()} Model Test {metric_type.capitalize()}')
        plt.xlabel('Round')
        plt.ylabel(metric_type.capitalize())
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
        if metric_type == 'accuracy':
            plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        
        plt.tight_layout()
        plt.show()