% I adapted Mianxin's code to plot h values from a fitted Ising model.
% -Adam Craig, 2025-01-02
% requires spm12 MATLAB library:
% https://www.fil.ion.ucl.ac.uk/spm/software/spm12/
% Add all the folders and subfolders of it to the path.

clear;
atlas_directory = 'glasser_atlas';
data_directory = 'data';
figure_directory = 'figures';

atlas_file = [atlas_directory filesep 'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii'];
atlas = ft_read_cifti(atlas_file);
mask = ~isnan(atlas.indexmax);

data_file = [data_directory filesep 'table_h_group_threshold_1_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000.dlm'];
data_table = readtable(data_file,'FileType','text','delimiter','\t');
feature = data_table.h;
feature_title = 'group model h_i values for threshold=1';
feature_file_prefix = 'group_h_1';

% Map from ROIs back to gray-ordinate voxels.
feature_expanded = nan( size(mask) );
feature_expanded_masked = feature( atlas.indexmax(mask) );
feature_expanded(mask) = feature_expanded_masked;

halfway_point = numel(feature_expanded)/2;
color_bounds = [min(feature_expanded_masked) max(feature_expanded_masked)];

[fig_handle_l, plot_handle_l, light_handle_l] = make_hemisphere_plot(feature_expanded, atlas_directory, feature_title, color_bounds, 'L');
left_file_prefix = [feature_file_prefix '_left'];
saveas(fig_handle_l, [figure_directory filesep left_file_prefix '.fig'], 'fig')
save_rotated(fig_handle_l, plot_handle_l, figure_directory, left_file_prefix)

[fig_handle_r, plot_handle_r, light_handle_r] = make_hemisphere_plot(feature_expanded, atlas_directory, feature_title, color_bounds, 'R');
right_file_prefix = [feature_file_prefix '_right'];
saveas(fig_handle_r, [figure_directory filesep right_file_prefix '.fig'], 'fig')
save_rotated(fig_handle_r, plot_handle_r, figure_directory, right_file_prefix)
