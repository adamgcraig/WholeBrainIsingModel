% I adapted Mianxin's code to plot h values from a fitted Ising model.
% -Adam Craig, 2025-01-02
% requires spm12 MATLAB library:
% https://www.fil.ion.ucl.ac.uk/spm/software/spm12/
% Add all the folders and subfolders of it to the path.

% clear;
atlas_directory = 'glasser_atlas';
data_directory = 'E:\\ising_model_results_daai';
figure_directory = 'figures';

atlas_file = [atlas_directory filesep 'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii'];
atlas = ft_read_cifti(atlas_file);
mask = ~isnan(atlas.indexmax);

% data_file = [data_directory filesep 'table_h_group_threshold_1_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000.dlm'];
% data_table = readtable(data_file,'FileType','text','delimiter','\t');
% feature = data_table.h;
% feature_title = 'group model h_i values for threshold=1';
% feature_file_prefix = 'group_h_1';
% cmap = 'hot';

% data_file = [data_directory filesep 'mean_thickness.dlm'];
% data_table = readtable(data_file,'FileType','text','delimiter','\t');
% feature = ( data_table.Var4 - mean(data_table.Var4) )/std(data_table.Var4);
% feature_title = 'mean thickness';
% feature_file_prefix = 'mean_thickness';
% cmap = 'winter';

% data_file = [data_directory filesep 'mean_myelination.dlm'];
% data_table = readtable(data_file,'FileType','text','delimiter','\t');
% feature = ( data_table.Var4 - mean(data_table.Var4) )/std(data_table.Var4);
% feature_title = 'mean myelination';
% feature_file_prefix = 'mean_myelination';
% cmap = 'spring';

% data_file = [data_directory filesep 'mean_curvature.dlm'];
% data_table = readtable(data_file,'FileType','text','delimiter','\t');
% feature = ( data_table.Var4 - mean(data_table.Var4) )/std(data_table.Var4);
% feature_title = 'mean curvature';
% feature_file_prefix = 'mean_curvature';
% cmap = 'summer';

% data_file = [data_directory filesep 'mean_sulcus_depth.dlm'];
% data_table = readtable(data_file,'FileType','text','delimiter','\t');
% feature = ( data_table.Var4 - mean(data_table.Var4) )/std(data_table.Var4);
% feature_title = 'mean sulcus depth';
% feature_file_prefix = 'mean_selcus_depth';
% cmap = 'autumn';

% color_bounds = [min(feature_expanded_masked) max(feature_expanded_masked)];

% data_file = [data_directory filesep 'group_h_threshold_1.dlm'];
% data_table = readtable(data_file,'FileType','text','delimiter','\t');
% feature = data_table.color;
% color_limits = [-1.29 1.23];
% cmap = 'hot';

% data_file = [data_directory filesep 'group_h_threshold_1.dlm'];
% data_table = readtable(data_file,'FileType','text','delimiter','\t');
% feature = data_table.color;
% color_limits = [-20.21 6.76];
% cmap = 'hot';

% figure_file_name = 'group_h_mean_std_1.png';
% data_file = [data_directory filesep 'group_h_threshold_1.dlm'];
% data_table = readtable(data_file,'FileType','text','delimiter','\t');
% feature = data_table.color;
% feature = ( feature - mean(feature) )/std(feature);
% color_limits = [min(feature) max(feature)];
% cmap = 'hot';

% figure_file_name = 'group_h_mean_std_0.png';
% data_file = [data_directory filesep 'group_h_threshold_0.dlm'];
% data_table = readtable(data_file,'FileType','text','delimiter','\t');
% feature = data_table.color;
% feature = ( feature - mean(feature) )/std(feature);
% color_limits = [min(feature) max(feature)];
% cmap = 'hot';

% feature = 'thickness';
% cmap = 'winter';

% feature = 'myelination';
% cmap = 'spring';

% feature = 'curvature';
% cmap = 'summer';

% feature = 'sulcus_depth';
% cmap = 'autumn';

% data_file = [data_directory filesep 'group_' feature '.dlm'];
% data_table = readtable(data_file,'FileType','text','delimiter','\t');
% feature = data_table.color;
% color_limits = [-0.358 0.358];
% cmap = 'hot';

% figure_file_name = 'std_myelination_hot.png';
% feature = 'myelination';
% data_file = [data_directory filesep 'individual_std_' feature '.dlm'];
% data_table = readtable(data_file,'FileType','text','delimiter','\t');
% feature = data_table.color;
% color_limits = [min(feature) max(feature)];
% cmap = 'hot';

% figure_file_name = 'std_h_1_hot.png';
% data_file = [data_directory filesep 'h_std_all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps_40000.dlm'];
% data_table = readtable(data_file,'FileType','text','delimiter','\t');
% feature = data_table.color;
% color_limits = [min(feature) max(feature)];
% cmap = 'hot';

% data_file = [data_directory filesep 'individual_corr_' feature '_all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps_40000.dlm'];
% data_table = readtable(data_file,'FileType','text','delimiter','\t');
% feature = data_table.color;
% color_limits = [-0.251 0.452];
% cmap = 'gray';
% cmap = 'hot';

% figure_file_name = 'mean_myelination_individual_correlation_sig_only_autumn.png';
% feature_name = 'myelination';
% corr_file = [data_directory filesep 'individual_corr_' feature_name '_all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps_40000.dlm'];
% corr_table = readtable(corr_file,'FileType','text','delimiter','\t');
% feature = corr_table.color;
% sig_file = [data_directory filesep 'individual_corr_is_sig_' feature_name '_all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps_40000.dlm'];
% is_sig = readtable(sig_file,'FileType','text','delimiter','\t').color > 0;
% feature_min = min(feature);
% feature_max = max(feature);
% color_limits = [min(feature) feature_max];
% feature(~is_sig) = feature_min-1;
% cmap = 'autumn';

% figure_file_name = 'linear_model_h_correlation_hot.png';
% feature_name = 'myelination';
% corr_file = [data_directory filesep 'linear_model_h_corr.dlm'];
% corr_table = readtable(corr_file,'FileType','text','delimiter','\t');
% feature = corr_table.color;
% feature_min = min(feature);
% feature_max = max(feature);
% color_limits = [feature_min feature_max];
% cmap = 'hot';

% figure_file_name = 'linear_model_h_correlation_sig_only_autumn.png';
% feature_name = 'myelination';
% corr_file = [data_directory filesep 'linear_model_h_corr.dlm'];
% corr_table = readtable(corr_file,'FileType','text','delimiter','\t');
% feature = corr_table.color;
% is_not_sig = corr_table.size >= ( 0.05/numel(corr_table.size) );
% disp( nnz(is_not_sig) )
% feature_min = min(feature);
% feature_max = max(feature);
% feature(is_not_sig) = feature_min-1;
% color_limits = [feature_min feature_max];
% cmap = 'autumn';

% feature_name = 'thickness';
% feature_name = 'myelination';
% feature_name = 'curvature';
% feature_name = 'sulcus_depth';
% figure_file_name = ['mean_' feature_name '_individual_correlation_sig_only_autumn.png'];
% corr_file = [data_directory filesep 'individual_corr_' feature_name '_all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps_40000.dlm'];
% corr_table = readtable(corr_file,'FileType','text','delimiter','\t');
% feature = corr_table.color;
% sig_file = [data_directory filesep 'individual_corr_is_sig_' feature_name '_all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps_40000.dlm'];
% is_sig = readtable(sig_file,'FileType','text','delimiter','\t').color > 0;
% feature_min = min(feature);
% feature_max = max(feature);
% color_limits = [min(feature) feature_max];
% feature(~is_sig) = feature_min-1;
% cmap = 'autumn';

% data_table = readtable(data_file,'FileType','text','delimiter','\t');
% feature = data_table.color;
% color_limits = [-0.251 0.452];
% cmap = 'gray';
% cmap = 'hot';

% data_file = [data_directory filesep 'individual_corr_is_sig_' feature '_all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps_40000.dlm'];
% data_table = readtable(data_file,'FileType','text','delimiter','\t');
% feature = data_table.color;
% color_limits = [0 1];
% cmap = 'gray';

% feature_name = 'thickness';
% feature_name = 'myelination';
% feature_name = 'curvature';
% feature_name = 'sulcus_depth';
% feature_name = 'group_h';
% feature_name = 'individual_h';
feature_name = 'h_pred_corr';
mean_or_std = 'mean';
% mean_or_std = 'std';
% cmap = 'autumn';
cmap = 'hot';
figure_file_name = [mean_or_std '_' feature_name '_' cmap '.png'];
if strcmp(feature_name, 'group_h')
    feature_table_file = [data_directory filesep 'h_mean_std_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000.dlm'];
elseif strcmp(feature_name, 'individual_h')
    feature_table_file = [data_directory filesep 'h_mean_std_group_threshold_1_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_68_param_updates_10000_individual_updates_45000.dlm'];
elseif strcmp(feature_name, 'h_pred_corr')
    feature_table_file = [data_directory filesep 'lstsq_corr_all_h_ising_model_light_group_threshold_1_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_68_param_updates_10000_individual_updates_45000.dlm'];
else
    feature_table_file = [data_directory filesep 'mean_std_' feature_name '.dlm'];
end
feature_table = readtable(feature_table_file,'FileType','text','delimiter','\t');
if strcmp(mean_or_std, 'mean')
    feature = feature_table.color;
else
    feature = feature_table.size;
end
feature_min = min(feature);
feature_max = max(feature);
color_limits = [feature_min feature_max];

% feature_name = 'thickness';
% % feature_name = 'myelination';
% % feature_name = 'curvature';
% % feature_name = 'sulcus_depth';
% mean_or_std = 'mean';
% % mean_or_std = 'std';
% sig_only_or_all = 'all';
% % sig_only_or_all = 'sig_only';
% cmap = 'autumn';
% % cmap = 'hot';
% figure_file_name = [mean_or_std '_' feature_name '_' sig_only_or_all '_' cmap '.png'];
% feature_table_file = [data_directory filesep 'mean_std_' feature_name '.dlm'];
% feature_table = readtable(feature_table_file,'FileType','text','delimiter','\t');
% if strcmp(mean_or_std, 'mean')
%     feature = feature_table.color;
% else
%     feature = feature_table.size;
% end
% sig_file = [data_directory filesep 'individual_corr_is_sig_' feature_name '_all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps_40000.dlm'];
% is_sig = readtable(sig_file,'FileType','text','delimiter','\t').color > 0;
% feature_min = min(feature);
% feature_max = max(feature);
% color_limits = [min(feature) feature_max];
% feature(~is_sig) = feature_min-1;

shinyness = 'dull';

% Map from ROIs back to gray-ordinate voxels.
feature_expanded = nan( size(mask) );
feature_expanded_masked = feature( atlas.indexmax(mask) );
feature_expanded(mask) = feature_expanded_masked;

halfway_point = numel(feature_expanded)/2;
feature_expanded_l = feature_expanded(1:halfway_point);
feature_expanded_r = feature_expanded( (halfway_point+1):end );

show_hemisphere_l.cdata = feature_expanded_l;
show_hemisphere_r.cdata = feature_expanded_r;

surf_hemisphere_l = gifti([atlas_directory filesep 'Q1-Q6_RelatedParcellation210.' 'L' '.midthickness_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii']);
surf_hemisphere_r = gifti([atlas_directory filesep 'Q1-Q6_RelatedParcellation210.' 'R' '.midthickness_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii']);

fig_left = 0;
fig_bottom = 0;
fig_width = 1750;
fig_height = 1000;
% upper_bottom = fig_bottom + fig_height/2;
% lower_bottom = fig_bottom;
% left_left = fig_left;
% center_left_left = fig_left + fig_width/3;
% center_right_left = fig_left + fig_width/2;
% right_left = fig_left + 2*fig_width/3;
% side_dims = [fig_width/3 fig_height/2];
% center_dims = [fig_width/6 fig_height];
delay = 5;

fig_handle = figure('Position',[fig_left fig_bottom fig_width fig_height]);
% pause(delay)

tiledlayout(2,3,'TileSpacing','none')
% pause(delay)

% In the upper left and right subplots, we rotate to see one side.

% axes_upper_left = subplot('Position',[left_left upper_bottom side_dims]);
axes_upper_left = nexttile(1,[1 1]);
patch_handle_upper_left = plot_gifti(surf_hemisphere_l, show_hemisphere_l, axes_upper_left, cmap, shinyness);
upper_left_transform = hgtransform;
patch_handle_upper_left.Parent = upper_left_transform;
upper_left_rotation = makehgtform('xrotate',-pi/2) * makehgtform('zrotate',pi/2);
upper_left_transform.Matrix = upper_left_rotation * upper_left_transform.Matrix;
clim(color_limits);
% pause(delay)

% The right is the mirror image of the left, so it needs the inverse.
% axes_upper_right = subplot('Position',[right_left upper_bottom side_dims]);
axes_upper_right = nexttile(3,[1 1]);
patch_handle_upper_right = plot_gifti(surf_hemisphere_r, show_hemisphere_r, axes_upper_right, cmap, shinyness);
upper_right_transform = hgtransform;
patch_handle_upper_right.Parent = upper_right_transform;
upper_right_rotation = makehgtform('yrotate',pi) * upper_left_rotation;
upper_right_transform.Matrix = upper_right_rotation * upper_right_transform.Matrix;
clim(color_limits);
% pause(delay)

% In the lower left and right subplots, we add a horizontal flip.

% axes_lower_left = subplot('Position',[left_left lower_bottom side_dims]);
axes_lower_left = nexttile(4,[1 1]);
patch_handle_lower_left = plot_gifti(surf_hemisphere_l, show_hemisphere_l, axes_lower_left, cmap, shinyness);
lower_left_transform = hgtransform;
patch_handle_lower_left.Parent = lower_left_transform;
lower_left_rotation = upper_right_rotation;
lower_left_transform.Matrix = lower_left_rotation * lower_left_transform.Matrix;
clim(color_limits);
% pause(delay)

% axes_lower_right = subplot('Position',[right_left lower_bottom side_dims]);
axes_lower_right = nexttile(6,[1 1]);
patch_handle_lower_right = plot_gifti(surf_hemisphere_r, show_hemisphere_r, axes_lower_right, cmap, shinyness);
lower_right_transform = hgtransform;
patch_handle_lower_right.Parent = lower_right_transform;
lower_right_transform.Matrix = upper_left_rotation * lower_right_transform.Matrix;
clim(color_limits);
% pause(delay)

% The middle plots are the unrotated default angle, the top view.

% axes_center_left = subplot('Position',[center_left_left lower_bottom center_dims]);
axes_center = nexttile(2,[2 1]);
% patch_handle_center_left = plot_gifti(surf_hemisphere_l, show_hemisphere_l, axes_center_left);
patch_handle_center_left = plot_gifti(surf_hemisphere_l, show_hemisphere_l, axes_center, cmap, shinyness);
clim(color_limits);
% pause(delay)

% axes_center_right = subplot('Position',[center_right_left lower_bottom center_dims]);
% axes_center_right = nexttile(4,[2 1]);
% patch_handle_center_right = plot_gifti(surf_hemisphere_r, show_hemisphere_r, axes_center_right);
patch_handle_center_right = plot_gifti(surf_hemisphere_r, show_hemisphere_r, axes_center, cmap, shinyness);
clim(color_limits);
pause(delay)


axis([axes_center axes_upper_left axes_lower_left axes_upper_right axes_lower_right],'equal')

colorbar_handle = colorbar('south');
colorbar_handle.FontSize = 30;
colorbar_handle.Position = colorbar_handle.Position.*[1 1 2 2];
colorbar_handle.Position = colorbar_handle.Position - [0 0.1 0.5*colorbar_handle.Position(3) 0];
colorbar_handle.AxisLocation = 'out';
colorbar_handle.Color = 'black';
% colorbar_handle.Color = 'white';

% layout_handle.Padding = 'tight';
% layout_handle.TileSpacing = 'none';
% axis(axes_upper_left,'equal');
% axis(axes_lower_left,'equal');
% axis(axes_upper_right,'equal');
% axis(axes_lower_right,'equal');
% axis(axes_center,'equal');
% axis(axes_center_left,'equal');
% axis(axes_center_right,'equal');

exportgraphics(fig_handle,figure_file_name,'ContentType','image','BackgroundColor','current')