#!/bin/bash

# Define the directories and other constants
hcp_dir=/mnt
subj_dir=/mnt/outputs
connectome=/mnt/connectome

[[ ! -e $subj_dir ]] && mkdir -p $subj_dir

schaefer_dir=/opt/HCP/schaefer
Glasser_dir=/opt/HCP/Glasser_parcellations
HCPpipeline_dir=/opt//HCP/HCPpipelines-4.7.0
mrtrixe3_dir=/opt/mrtrix3/share/mrtrix3/labelconvert
nthreads=12

subj=$1

echo "Processing subject $subj"

diff_dir=$hcp_dir/$subj/T1w/Diffusion
[[ ! -e $subj_dir/$subj/structural_connectivity ]] && mkdir -p $subj_dir/$subj/structural_connectivity

[[ ! -e $connectome/$subj ]] & mkdir -p  $connectome/$subj
output_dir=$subj_dir/$subj/structural_connectivity

echo -e "Processing subject $subj on" $(date)

## -------------------------------------------------------------------------------------------------------------------------------------------
## DIFFUSION IMAGE PROCESSING

# Convert the HCP-derived preprocessed diffusion data to MRtrix format
mrconvert $diff_dir/data.nii.gz $output_dir/DWI.mif -fslgrad $diff_dir/bvecs $diff_dir/bvals -datatype float32 -strides 0,0,0,1 -quiet -force

# Generate a mean b=0 image for visualization purposes
dwiextract $output_dir/DWI.mif - -bzero | mrmath - mean $output_dir/meanb0.mif -axis 3

# Generate a basis response function, specific to the subject's data
dwi2response dhollander $output_dir/DWI.mif $output_dir/RF_WM_dhollander.txt $output_dir/RF_GM_dhollander.txt $output_dir/RF_CSF_dhollander.txt -voxels $output_dir/RF_voxels_dhollander.mif -nthreads ${nthreads} -force

# Create Fiber Orientation Densities (i.e., perform Multi-Shell, Multi-Tissue Constrained Spherical Deconvolution)
dwi2fod msmt_csd $output_dir/DWI.mif $output_dir/RF_WM_dhollander.txt $output_dir/RF_WM_FOD_dhollander.mif $output_dir/RF_GM_dhollander.txt $output_dir/RF_GM_FOD_dhollander.mif $output_dir/RF_CSF_dhollander.txt $output_dir/RF_CSF_FOD_dhollander.mif -mask $diff_dir/nodif_brain_mask.nii.gz -nthreads ${nthreads} -force

# Combine the above FODs into one image, for visualization purposes
mrconvert $output_dir/RF_WM_FOD_dhollander.mif - -coord 3 0 | mrcat $output_dir/RF_CSF_FOD_dhollander.mif $output_dir/RF_GM_FOD_dhollander.mif - $output_dir/combined_FODs.mif -axis 3 -nthreads ${nthreads}
# Keep track of time
echo -e "Diffusion image processing completed on" $(date)

## -------------------------------------------------------------------------------------------------------------------------------------------
## STRUCTURAL IMAGE PROCESSING

# Segment the anatomical image into tissue types, appropriate for Anatomically-Constrained Tractography
SGE_ROOT="" 5ttgen fsl $hcp_dir/$subj/T1w/T1w_acpc_dc_restore_brain.nii.gz $output_dir/5TT_in_T1w_space.mif -premasked -force

# Collapse the multi-tissue image into a 3D greyscale image for visualisation
# 5tt2vis $output_dir/5TT_in_T1w_space.mif $output_dir/5TT_vis.mif

# Keep track of time
echo -e "Structural image processing completed on" $(date)

## -------------------------------------------------------------------------------------------------------------------------------------------
## TRACTOGRAPHY

# Generate the initial tractogram
tckgen $output_dir/RF_WM_FOD_dhollander.mif $output_dir/tracks_10M.tck -algorithm iFOD2 -act $output_dir/5TT_in_T1w_space.mif -backtrack -crop_at_gmwmi -seed_dynamic $output_dir/RF_WM_FOD_dhollander.mif -maxlength 300 -select 10M -cutoff 0.06 -nthreads ${nthreads} -force


schaefer_atlas="Schaefer2018_200Parcels_7Networks_order"

export SUBJECTS_DIR=$hcp_dir/$subj/T1w

for hemi in lh rh;
do
# Considering how the HCP data release doesn't have an fsaverage subject in their $SUBJECTS_DIR I'll need to be creative
mri_surf2surf --hemi $hemi --srcsubject ../../../../../../../opt/freesurfer/subjects/fsaverage5 --trgsubject $subj --sval-annot $schaefer_dir/FreeSurfer5.3/fsaverage5/label/${hemi}.${schaefer_atlas}.annot --tval $output_dir/${hemi}.${schaefer_atlas}.annot
done

# Generate the Schaefer2018 parcellation in volume space
# Getting creative one more time
[[ ! -e $subj_dir/$subj/mri ]] && mkdir -p $subj_dir/$subj/mri
[[ ! -e $subj_dir/$subj/surf ]] && mkdir -p $subj_dir/$subj/surf
[[ ! -e $subj_dir/$subj/label ]] && mkdir -p $subj_dir/$subj/label

mv $output_dir/*h.${schaefer_atlas}.annot $subj_dir/$subj/label/

cp -f $hcp_dir/$subj/T1w/$subj/surf/*h.pial $hcp_dir/$subj/T1w/$subj/surf/*h.white $subj_dir/$subj/surf/
cp -f $hcp_dir/$subj/T1w/$subj/mri/ribbon.mgz $hcp_dir/$subj/T1w/$subj/mri/aseg.mgz $hcp_dir/$subj/T1w/$subj/mri/wm.mgz $subj_dir/$subj/mri/

mri_binarize --i $subj_dir/$subj/mri/wm.mgz --o $output_dir/wm_mask.nii.gz --min 0.5 --dilate 1 --erode 1


export SUBJECTS_DIR=$subj_dir

mri_aparc2aseg --s $subj --o $output_dir/${schaefer_atlas}.mgz --annot $schaefer_atlas

# Convert the labels of the atlas parcellation to a format that MRtrix understands
labelconvert $output_dir/${schaefer_atlas}.mgz $schaefer_dir/project_to_individual/${schaefer_atlas}_LUT.txt $schaefer_dir/freeview_lut/${schaefer_atlas}.txt $output_dir/${schaefer_atlas}_parcels.mif -force

mrconvert $output_dir/DWI.mif $output_dir/DWI.nii.gz

flirt -in $output_dir/wm_mask.nii.gz \
     -ref $output_dir/DWI.nii.gz \
     -applyxfm -usesqform \
     -out $output_dir/wm_mask_resampled.nii.gz

	 
cp -f $hcp_dir/$subj/T1w/Diffusion/bvals $hcp_dir/$subj/T1w/Diffusion/bvecs $output_dir

## -------------------------------------------------------------------------------------------------------------------------------------------

# Get hemispheric gii versions of the dlabel Glasser atlas
wb_command -cifti-separate /opt/HCP/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii COLUMN -label CORTEX_LEFT $subj_dir/$subj/label/L.Glasser_cortical.label.gii -label CORTEX_RIGHT $subj_dir/$subj/label/R.Glasser_cortical.label.gii

for hemi in L R;
do
	# Resample the Glasser label files from fs_32k space to fsaverage space
	wb_command -label-resample $subj_dir/$subj/label/${hemi}.Glasser_cortical.label.gii $HCPpipeline_dir/global/templates/standard_mesh_atlases/${hemi}.sphere.32k_fs_LR.surf.gii $HCPpipeline_dir/global/templates/standard_mesh_atlases/fs_${hemi}/fs_${hemi}-to-fs_LR_fsaverage.${hemi}_LR.spherical_std.164k_fs_${hemi}.surf.gii BARYCENTRIC $subj_dir/$subj/label/${hemi}.Glasser_cortical.fsaverage164.label.gii

	hemi2=$(echo "${hemi}"h | tr '[:upper:]' '[:lower:]')
	
	echo ${hemi2}

	# Using FreeSurfer, convert the gii files to annot files
	mris_convert --annot $subj_dir/$subj/label/${hemi}.Glasser_cortical.fsaverage164.label.gii $HCPpipeline_dir/global/templates/standard_mesh_atlases/fs_${hemi}/fs_${hemi}-to-fs_LR_fsaverage.${hemi}_LR.spherical_std.164k_fs_${hemi}.surf.gii $subj_dir/$subj/label/${hemi2}.Glasser_cortical_fsaverage.annot

	# Now perform the mri_surf2surf command as in the Schaefer case above, to send the annot files from fsaverage to subject space
	export SUBJECTS_DIR=$hcp_dir/$subj/T1w

	mri_surf2surf --hemi ${hemi2} --srcsubject ../../../../../../../opt/freesurfer/subjects/fsaverage --trgsubject $subj --sval-annot $subj_dir/$subj/label/${hemi2}.Glasser_cortical_fsaverage.annot --tval $subj_dir/$subj/label/${hemi2}.Glasser_cortical_tmp.annot
done	

./process_annotations $subj_dir $subj

# Generate the corresponding parcellation volume in subject space
export SUBJECTS_DIR=$subj_dir

mri_aparc2aseg --s $subj --o $output_dir/Glasser_atlas.mgz --annot Glasser_cortical

# Convert the labels of the atlas parcellation to a format that MRtrix understands
labelconvert $output_dir/Glasser_atlas.mgz $mrtrixe3_dir/hcpmmp1_original.txt $mrtrixe3_dir/hcpmmp1_ordered.txt $output_dir/Glasser_atlas_cortical_parcels.mif -force

mrconvert $output_dir/Glasser_atlas_cortical_parcels.mif $output_dir/Glasser_atlas_cortical_parcels.nii.gz

mrconvert $output_dir/${schaefer_atlas}_parcels.mif $output_dir/${schaefer_atlas}_parcels.nii.gz

# Perform the actual tractography, as above
tck2connectome -scale_invnodevol -assignment_radial_search 2 $output_dir/tracks_10M.tck $output_dir/Glasser_atlas_cortical_parcels.mif $connectome/$subj/${subj}_Glasser.csv -nthreads ${nthreads} -force

schaefer_atlas="Schaefer2018_200Parcels_7Networks_order"
tcksift2 $output_dir/tracks_10M.tck $output_dir/RF_WM_FOD_dhollander.mif $output_dir/tracks_10M_sift2.txt -act $output_dir/5TT_in_T1w_space.mif -force

tck2connectome -scale_invnodevol -assignment_radial_search 2 -tck_weights_in $output_dir/tracks_10M_sift2.txt  $output_dir/tracks_10M.tck $output_dir/Glasser_atlas_cortical_parcels.mif $connectome/$subj/${subj}_Glasser_sift2.csv -nthreads ${nthreads} -force

tck2connectome -scale_invnodevol -assignment_radial_search 2 $output_dir/tracks_10M.tck $output_dir/${schaefer_atlas}_parcels.mif $connectome/$subj/${subj}_${schaefer_atlas}.csv -nthreads ${nthreads} -force

tck2connectome -scale_invnodevol -assignment_radial_search 2 -tck_weights_in $output_dir/tracks_10M_sift2.txt  $output_dir/tracks_10M.tck $output_dir/${schaefer_atlas}_parcels.mif $connectome/$subj/${subj}_${schaefer_atlas}_sift2.csv -nthreads ${nthreads} -force


echo -e "Tractography completed on" $(date)
echo -e "Subject ${subj} is all done!"	



