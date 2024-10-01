import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import matplotlib.pyplot as plt

subject_ids = ['100206']
fMRI_files = [
    '/kaggle/input/hcp-100206/100206_tfMRI_RELATIONAL_RL.nii/100206_tfMRI_RELATIONAL_RL.nii' 
]
event_directories = [
    '/kaggle/input/hcp-100206/EVs_100206'  
]
atlas_filename = '/kaggle/input/aal-nii-atlas/AAL.nii' 
output_dir = '/kaggle/working/'  
os.makedirs(output_dir, exist_ok=True)

tr = 0.72  

for fMRI_file, event_dir in zip(fMRI_files, event_directories):
    print(f"Processing fMRI file: {fMRI_file}")

    fmri_img = nib.load(fMRI_file)
    print(f"fMRI data shape: {fmri_img.shape}")

    relational_file = os.path.join(event_dir, 'relation.txt')
    match_file = os.path.join(event_dir, 'match.txt')

    atlas_img = nib.load(atlas_filename)
    print(f"AAL atlas shape: {atlas_img.shape}")

    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=True, memory='nilearn_cache')
    time_series = masker.fit_transform(fmri_img)
    print(f"Extracted time series shape: {time_series.shape}")

    relational_events = pd.read_csv(relational_file, sep='\t', header=None, names=['onset', 'duration', 'condition'])
    print(f"Relational Events DataFrame shape: {relational_events.shape}")

    match_events = pd.read_csv(match_file, sep='\t', header=None, names=['onset', 'duration', 'condition'])
    print(f"Match Events DataFrame shape: {match_events.shape}")

    relational_events.rename(columns={'condition': 'trial_type'}, inplace=True)
    match_events.rename(columns={'condition': 'trial_type'}, inplace=True)
    match_events['trial_type'] = 'Match'
    relational_events['trial_type'] = 'Relational'

    n_scans = fmri_img.shape[-1]  
    frame_times = np.arange(n_scans) * tr  

    all_events = pd.concat([relational_events, match_events])

    full_time_grid = np.zeros(n_scans)  

    for _, event in all_events.iterrows():
        onset = int(np.floor(event['onset'] / tr))
        offset = int(np.ceil((event['onset'] + event['duration']) / tr))
        if event['trial_type'] == 'Relational':
            full_time_grid[onset:offset] = 1  
        elif event['trial_type'] == 'Match':
            full_time_grid[onset:offset] = 2  

    rest_indices = np.where(full_time_grid == 0)[0]
    rest_events = pd.DataFrame({
        'onset': rest_indices * tr,
        'duration': [tr] * len(rest_indices),
        'trial_type': ['Rest'] * len(rest_indices)
    })

    print(f"Number of rest events: {len(rest_events)}")

    final_events = pd.concat([all_events, rest_events])

    conditions = ['Relational', 'Match', 'Rest']

    for condition_name in conditions:
        condition_events = final_events[final_events['trial_type'] == condition_name]

       # print(f"\nProcessing condition: {condition_name} with {condition_events.shape[0]} events")
        
        for _, event in condition_events.iterrows():
            onset_time = event['onset']
            end_time = event['onset'] + event['duration']
           # print(f"From {onset_time:.2f} sec to {end_time:.2f} sec")

        condition_time_series = []
        for onset, duration in zip(condition_events['onset'], condition_events['duration']):
            start_frame = int(np.floor(onset / tr))
            end_frame = int(np.ceil((onset + duration) / tr))
            condition_time_series.extend(time_series[start_frame:end_frame])

        condition_time_series = np.array(condition_time_series)
        #print(f"Condition time series shape for {condition_name}: {condition_time_series.shape}")

        

        correlation_measure = ConnectivityMeasure(kind='correlation')
        fc_matrix = correlation_measure.fit_transform([condition_time_series])[0]
        print(f"Functional connectivity matrix shape for {condition_name}: {fc_matrix.shape}")

        fc_matrix_filename = os.path.join(output_dir, f'fc_matrix_{os.path.basename(fMRI_file)}_{condition_name}.csv')
        np.savetxt(fc_matrix_filename, fc_matrix, delimiter=',')
        print(f"Saved FC matrix for {condition_name} to {fc_matrix_filename}")

        plt.imshow(fc_matrix, cmap='coolwarm', vmax=1, vmin=-1)
        plt.colorbar()
        plt.title(f'FC Matrix - {condition_name}')
        plt.savefig(os.path.join(output_dir, f'fc_matrix_{condition_name}.png'))
        plt.close()
        
        print(fc_matrix[:5,:5])
