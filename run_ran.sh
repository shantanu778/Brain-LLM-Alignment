#!/bin/bash
corpus_type=$1  # 'char' or 'word'
echo "Processing Corrupt Version: $corpus_type"
subject=('UTS01' 'UTS02' 'UTS03' 'UTS04' 'UTS05' 'UTS06' 'UTS07' 'UTS08' 'UTS09')
rois=('G_Fusiform-4.nii' 'G_Temporal_Mid-3.nii' 'G_Temporal_Mid-4.nii' 'G_ParaHippocampal-1.nii' 'G_Temporal_Inf-4.nii' 'G_Supp_Motor_Area-2.nii' 'G_Supp_Motor_Area-3.nii' 'G_Insula-anterior-2.nii' 'G_Insula-anterior-3.nii' 'G_Insula-anterior-1.nii' 'G_Angular-2.nii' 'N_Thalamus-4.nii' 'G_Occipital_Inf-1.nii' 'G_Temporal_Sup-4.nii' 'G_Frontal_Inf_Orb-1.nii' 'S_Sup_Temporal-4.nii' 'G_Frontal_Inf_Tri-1.nii' 'G_Cingulum_Post-3.nii' 'S_Precentral-3.nii' 'G_SupraMarginal-7.nii' 'S_Precentral-4.nii' 'S_Sup_Temporal-2.nii' 'S_Sup_Temporal-3.nii' 'S_Sup_Temporal-1.nii' 'S_Inf_Frontal-2.nii' 'G_Precuneus-6.nii' 'G_Paracentral_Lobule-4.nii' 'N_Amygdala-1.nii' 'G_Frontal_Sup-2.nii' 'N_Putamen-2.nii' 'G_Hippocampus-2.nii' 'N_Putamen-3.nii')

# subject=('UTS01' 'UTS02')
# rois=('G_Fusiform-4.nii' 'G_Temporal_Mid-3.nii')

for subject in "${subject[@]}"; do
    for roi in "${rois[@]}"; do
        echo "Processing subject: $subject, ROI: $roi"
        python alignment/main.py -f Random-corruption-dirs/$corpus_type-equiv/gpt2/char/models/config.json \
                                  --sub $subject \
                                  --roi $roi 
    done
done
# subjects

