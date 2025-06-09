ECHO starting download
FOR /F "tokens=*" %%A in (concat_subjects.txt) do (
   ECHO %%A
   aws s3 cp s3://hcp-openaccess/HCP_1200/%%A/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz D:\hcp_from_s3_second_try\%%A_rest_1_lr.nii.gz
   aws s3 cp s3://hcp-openaccess/HCP_1200/%%A/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL.nii.gz D:\hcp_from_s3_second_try\%%A_rest_1_rl.nii.gz
   aws s3 cp s3://hcp-openaccess/HCP_1200/%%A/MNINonLinear/Results/rfMRI_REST2_LR/rfMRI_REST2_LR.nii.gz D:\hcp_from_s3_second_try\%%A_rest_2_lr.nii.gz
   aws s3 cp s3://hcp-openaccess/HCP_1200/%%A/MNINonLinear/Results/rfMRI_REST2_RL/rfMRI_REST2_RL.nii.gz D:\hcp_from_s3_second_try\%%A_rest_2_rl.nii.gz
)
PAUSE