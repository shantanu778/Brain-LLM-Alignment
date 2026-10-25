import json
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Train linear model to predict fMRI responses from LLM representations")
parser.add_argument('-m', '--model_type', type=str, required=True, help="Model Type")
parser.add_argument('-c', '--corpus_type', type=str, required=True, help="Coupus Type")
parser.add_argument('-l', '--layer', type=str, default="last", help="layer no or last for last hidden layer")
args = parser.parse_args()

if args.layer == "last":
    print(f"Processing correlations for {args.model_type} on {args.corpus_type} for layer {args.layer}")
else:
    args.layer = int(args.layer)
    print(f"Processing correlations for {args.model_type} on {args.corpus_type} for layer {args.layer}")


subject=['UTS01', 'UTS02', 'UTS03', 'UTS04', 'UTS05', 'UTS06', 'UTS07', 'UTS08', 'UTS09']

rois=['G_Fusiform-4.nii', 'G_Temporal_Mid-3.nii', 'G_Temporal_Mid-4.nii', 'G_ParaHippocampal-1.nii', 'G_Temporal_Inf-4.nii', 'G_Supp_Motor_Area-2.nii', 'G_Supp_Motor_Area-3.nii', 'G_Insula-anterior-2.nii', 'G_Insula-anterior-3.nii', 'G_Insula-anterior-1.nii', 'G_Angular-2.nii', 'N_Thalamus-4.nii', 'G_Occipital_Inf-1.nii', 'G_Temporal_Sup-4.nii', 'G_Frontal_Inf_Orb-1.nii', 'S_Sup_Temporal-4.nii', 'G_Frontal_Inf_Tri-1.nii', 'G_Cingulum_Post-3.nii', 'S_Precentral-3.nii', 'G_SupraMarginal-7.nii', 'S_Precentral-4.nii', 'S_Sup_Temporal-2.nii', 'S_Sup_Temporal-3.nii', 'S_Sup_Temporal-1.nii', 'S_Inf_Frontal-2.nii', 'G_Precuneus-6.nii', 'G_Paracentral_Lobule-4.nii', 'N_Amygdala-1.nii', 'G_Frontal_Sup-2.nii', 'N_Putamen-2.nii' , 'G_Hippocampus-2.nii','N_Putamen-3.nii']

print(f"Processing {len(rois)} ROIs for {len(subject)} subjects")

log = {
        'corpus_type': [],
        'model_type': [],
        'roi': [],
        'subject': [],
        'mean_corrs': [],
        'std': []
    }
for sub in subject:
    for roi in rois:
        # print(f"Processing subject: {sub}, ROI: {roi}, Model: {args.model_type}")
        if args.layer:
            with open(f'{args.corpus_type}/sub-{sub}/corrs_{sub}_{roi}_{args.model_type}_layer_{args.layer}.npy', 'rb') as f:
                corrs = np.load(f)
        else:
            with open(f'{args.corpus_type}/sub-{sub}/corrs_{sub}_{roi}_{args.model_type}.npy', 'rb') as f:
                corrs = np.load(f)

        log['corpus_type'].append(args.corpus_type)
        log['model_type'].append(args.model_type)
        log['roi'].append(roi)
        log['subject'].append(sub)
        log['mean_corrs'].append(f"{np.mean(corrs):.3f}")
        log['std'].append(f"{np.std(corrs):.3f}")




# Print the first 5 entries for each key to verify
df = pd.DataFrame.from_dict(log)

# ✅ Add empty rows after each subject using pandas
dfs_with_spacing = []

for sub in df['subject'].unique():
    # Get all rows for this subject
    sub_df = df[df['subject'] == sub].copy()
    dfs_with_spacing.append(sub_df)
    
    # Add empty row (create a DataFrame with one empty row)
    empty_row = pd.DataFrame([{col: '' for col in df.columns}])
    dfs_with_spacing.append(empty_row)

# Concatenate all DataFrames
df_spaced = pd.concat(dfs_with_spacing, ignore_index=True)

# Save to Excel with spacing
if args.layer:
    df_spaced.to_excel(f'{args.corpus_type}/{args.model_type}_layer_{args.layer}_correlations.xlsx', index=False)
else:
    df_spaced.to_excel(f'{args.corpus_type}/{args.model_type}_correlations.xlsx', index=False)
