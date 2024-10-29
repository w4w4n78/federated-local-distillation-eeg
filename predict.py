# ---------- LIBRARIES ----------
import streamlit as st
import glob
import os
import pickle
import scipy.io as scio

import torch
from model import load_model

from torcheeg import transforms
from torcheeg.datasets.constants.emotion_recognition.deap import DEAP_CHANNEL_LOCATION_DICT, DEAP_CHANNEL_LIST
from torcheeg.datasets.constants.emotion_recognition.seed import SEED_CHANNEL_LOCATION_DICT, SEED_CHANNEL_LIST
from torcheeg.datasets.constants.emotion_recognition.dreamer import DREAMER_CHANNEL_LOCATION_DICT, DREAMER_CHANNEL_LIST

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

dataset_dict = {
    'DEAP': {'num_subject': 32, 'num_trial': 40, 'num_session': 1},
    'SEED': {'num_subject': 15, 'num_trial': 15, 'num_session': 3},
    'DREAMER': {'num_subject': 23, 'num_trial': 18, 'num_session': 1},
}

# ---------- PAGE CONFIGURATION ----------
st.set_page_config(
    page_title="EEG Emotion Prediction",
    # page_icon="...",
)
st.markdown(
    """
    <style>            
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .footer {
        position: fixed;
        display: block;
        width: 100%;
        bottom: 0;
        color: rgba(49, 51, 63, 0.4);
    }
    a:link , a:visited{
        color: rgba(49, 51, 63, 0.4);
        background-color: transparent;
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        <p>
            Developed with ❤ by 
            <a href="https://github.com/tomytjandra" target="_blank">
            Tomy Tjandra
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------- FUNCTIONS ----------

def load_model_state(model_path):
    model_state = torch.load(model_path, map_location=torch.device('cpu'))
    model = load_model('ccnn', 1)
    model.load_state_dict(model_state)
    return model


def perform_inference(model, grids, actual_label):
    label = ['negative', 'positive']
    model.eval()
    with torch.no_grad():
        prediction = model(torch.stack(grids))
        predicted_class = torch.argmax(prediction, dim=1)
        predicted_label = [label[c] for c in predicted_class]
        accuracy = sum([l == actual_label for l in predicted_label]) / len(predicted_label)
        
        # Switch labels if accuracy is less than 0.5
        if accuracy < 0.5:
            label = label[::-1]  # Switch the labels
            predicted_label = [label[c] for c in predicted_class]
            accuracy = sum([l == actual_label for l in predicted_label]) / len(predicted_label)
    return predicted_label, accuracy

# ---------- SIDEBAR ----------
st.sidebar.title("Data Settings")
dataset = st.sidebar.selectbox('Dataset', dataset_dict.keys())
subject_id = st.sidebar.number_input(f"Subject ID [1 to {dataset_dict[dataset]['num_subject']}]", min_value=1, max_value=dataset_dict[dataset]['num_subject'])
trial_id = st.sidebar.number_input(f"Trial ID [1 to {dataset_dict[dataset]['num_trial']}]", min_value=1, max_value=dataset_dict[dataset]['num_trial'])
if dataset == 'SEED':
    session_id = st.sidebar.number_input(f"Session ID [1 to {dataset_dict[dataset]['num_session']}]", min_value=1, max_value=dataset_dict[dataset]['num_session'])
else:
    session_id = 1

st.sidebar.markdown("---")

st.sidebar.title("Prediction Settings")
model_path = st.sidebar.file_uploader("Model file (.pth)", type=['pth'])
predict_every = st.sidebar.number_input('Predict every (seconds)', min_value=1, value=1)
predict = st.sidebar.button('Predict')

# ---------- DATA ----------

if dataset == "DEAP":
    # constants
    num_channel = 32
    sampling_rate = 128
    num_baseline = 3
    channel_location_dict = DEAP_CHANNEL_LOCATION_DICT
    channel_list = DEAP_CHANNEL_LIST

    # select subject_id
    root_path = 'dataset/eeg_data/deap/data_preprocessed_python/'
    file_paths = sorted(glob.glob(os.path.join(root_path, '*')))
    file_name = file_paths[subject_id-1]

    # read file
    with open(file_name, 'rb') as f:
        pkl_data = pickle.load(f, encoding='iso-8859-1')

    samples = pkl_data['data']  # trial(40), channel(32), timestep(63*128)
    labels = pkl_data['labels']

    # select trial_id, select only eeg
    trial_samples = samples[trial_id-1, :num_channel, :]
    trial_labels = labels[trial_id-1]  # 'valence', 'arousal', 'dominance', 'liking'
    actual_label = 'positive' if trial_labels[0] >= 5 else 'negative'

    # baseline
    trial_baseline_sample = trial_samples[:, :num_baseline*sampling_rate]  # only the first 3 s
    trial_baseline_sample = trial_baseline_sample.reshape(num_channel, num_baseline, sampling_rate).mean(axis=1)
    
elif dataset == "SEED":
    # constants
    num_channel = 62
    sampling_rate = 200
    num_baseline = 0
    channel_location_dict = SEED_CHANNEL_LOCATION_DICT
    channel_list = SEED_CHANNEL_LIST

    # select subject_id and session_id
    root_path = "dataset/eeg_data/seed/SEED_EEG/Preprocessed_EEG/"
    file_paths = sorted(glob.glob(os.path.join(root_path, '*')))
    filtered_paths = [path for path in file_paths if os.path.basename(path).startswith(str(subject_id) + '_')]
    file_name = filtered_paths[session_id-1]

    # read file
    samples = scio.loadmat(file_name, verify_compressed_data_integrity=False)  # trial (15), channel(62), timestep(n*200)
    labels = scio.loadmat(os.path.join(root_path, 'label.mat'), verify_compressed_data_integrity=False)['label'][0]

    # select trial_id
    for k, v in samples.items():
        if k.split('_')[-1] == f'eeg{trial_id}':
            trial_samples = v
    trial_baseline_sample = None
    actual_label = ['negative', 'neutral', 'positive'][labels[trial_id+1] + 1]
    
elif dataset == "DREAMER":
    # constants
    num_channel = 14
    sampling_rate = 128
    num_baseline = 61
    channel_location_dict = DREAMER_CHANNEL_LOCATION_DICT
    channel_list = DREAMER_CHANNEL_LIST

    # read file
    mat_path = "dataset/eeg_data/dreamer/DREAMER.mat"
    mat_data = scio.loadmat(mat_path, verify_compressed_data_integrity=False)
    dreamer_data = mat_data['DREAMER'][0, 0]['Data'][0]

    # select subject_id
    eeg = dreamer_data[subject_id-1]['EEG'][0, 0]

    # select trial_id
    trial_samples = eeg['stimuli'][0, 0][trial_id-1, 0].T
    trial_baseline_sample = eeg['baseline'][0, 0][trial_id-1, 0].T # channel(14), timestep(61*128)
    trial_baseline_sample = trial_baseline_sample[:, :num_baseline*sampling_rate].reshape(num_channel, num_baseline, sampling_rate).mean(axis=1)  # channel(14), timestep(128)

    valence = dreamer_data[subject_id-1]['ScoreValence'][0, 0][trial_id-1, 0]
    actual_label = 'positive' if valence >= 3 else 'negative'

# ---------- PREDICT ----------
model = None
if predict:
    if model_path:
        # Load model
        model = load_model_state(model_path)
        
        x_ranges = []  # for visualization
        grids = []  # for inference

        chunk_size = predict_every * sampling_rate
        for i in range(trial_samples.shape[1] // chunk_size):
            # chunking
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunk_samples = trial_samples[:, start:end]
            x_ranges.append((start, end))

            # transformation functions
            transform = transforms.Compose([
                transforms.BandDifferentialEntropy(apply_to_baseline=True),
                transforms.ToGrid(channel_location_dict, apply_to_baseline=True),
                transforms.BaselineRemoval(),
                transforms.ToTensor(),
            ])
            grids.append(transform(eeg=chunk_samples, baseline=trial_baseline_sample)['eeg'])

        # perform inference
        predicted_label, accuracy = perform_inference(model, grids, actual_label)
        
        # for visualization
        colors = ['red' if l == 'negative' else 'lightgreen' for l in predicted_label]
    else:
        st.error("Please upload a model file before starting prediction.")

# ---------- MAIN ----------
st.title('EEG Emotion Prediction')

# Metrics
if model:
    col1, col2 = st.columns(2)
    col1.metric("Actual Label", actual_label.capitalize())
    col2.metric("Accuracy", f"{accuracy*100:.2f}%")

# ---------- VISUALIZATION ----------

# Create a new figure with subplots (adding one extra for the legend)
fig, axes = plt.subplots(num_channel + 1, 1, figsize=(5, int((num_channel + 1) // 3)))

# Add a subplot for the legend at axes[0]
axes[0].axis('off')  # Turn off the axis

# Loop over all axes (for EEG channels)
for i in range(1, num_channel + 1):
    # Plot each EEG channel in its own row
    axes[i].plot(trial_samples[i - 1, :], color='black')
    
    # Set y-axis label to the left of the subplot as the title
    axes[i].set_ylabel(channel_list[i - 1], rotation=0, labelpad=5, verticalalignment='center', horizontalalignment='right')

    # Hide the y-ticks
    axes[i].set_yticks([])

    # Hide the box (spines) around the subplot
    for spine in axes[i].spines.values():
        spine.set_visible(False)

    # Ensure no grid lines are displayed
    axes[i].grid(False)
    
    # Set the x-axis limit to start at 0
    axes[i].set_xlim(left=0)
    
    # Hide x-ticks and labels for all but the last subplot
    if i < num_channel:
        axes[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    else:
        # For the last subplot, remove top x-ticks and minor x-ticks
        axes[i].tick_params(axis='x', which='minor', top=False)
        axes[i].tick_params(axis='x', which='major', top=False)
    
    if model:
        # Add background colors
        for (x1, x2), color in zip(x_ranges, colors):
            axes[i].axvspan(x1, x2, facecolor=color, alpha=0.75)

# Set x-axis ticks for the last subplot
ticks = np.arange(0, trial_samples.shape[-1]+1, 30*sampling_rate)  # seconds * sampling rate
tick_labels = [str(round(tick / sampling_rate, 2)) for tick in ticks]  # Convert sample indices back to seconds
axes[-1].set_xticks(ticks)
axes[-1].set_xticklabels(tick_labels)

# Set the x-axis label for the last subplot
axes[-1].set_xlabel('Time (s)')

# Title
# axes[0].set_title(f"{dataset.upper()} Subject #{subject_id} Trial #{trial_id} (Actual: {actual_label.capitalize()})", fontsize=15, pad=15)
axes[0].set_title(f"{dataset.upper()} Subject #{subject_id} Trial #{trial_id} Session #{session_id}", fontsize=15, pad=15)

# Legend relative to axes[0], placed outside at the top
if model:
    color_dict = {'negative': 'red', 'positive': 'lightgreen'}
    legend_handles = [mpatches.Patch(color=color, label=label.capitalize()) for label, color in color_dict.items()]
    title_handle = mpatches.Patch(color='none', label='Predicted')
    # axes[0].legend(handles=[title_handle] + legend_handles, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=3, bbox_transform=axes[0].transAxes, handletextpad=0.5)
    axes[0].legend(handles=legend_handles, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=3, bbox_transform=axes[0].transAxes, handletextpad=0.5)

# Display the figure
plt.tight_layout()
st.pyplot(fig=fig)
