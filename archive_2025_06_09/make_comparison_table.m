data_directory = 'E:\\ising_model_results_daai';
indices = (1:360)';

feature_name = 'curvature';
feature_table_file = [data_directory filesep 'mean_std_' feature_name '.dlm'];
feature_table = readtable(feature_table_file,'FileType','text','delimiter','\t');
feature_mean = feature_table.color;
feature_std = feature_table.size;

name = feature_table.name;
x = feature_table.x;
y = feature_table.y;
z = feature_table.z;

[feature_std_sorted,feature_std_indices] = sort(feature_std, 'descend');
feature_std_rank = indices;
feature_std_rank(feature_std_indices) = indices;

h_pred_corr_file = [data_directory filesep 'lstsq_corr_all_h_ising_model_light_group_threshold_1_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_68_param_updates_10000_individual_updates_45000.dlm'];
h_pred_corr_table = readtable(h_pred_corr_file,'FileType','text','delimiter','\t');
h_pred_corr = h_pred_corr_table.color;
h_pred_p = h_pred_corr_table.size;

[h_pred_corr_sorted,h_pred_corr_indices] = sort(h_pred_corr, 'descend');
h_pred_corr_rank = indices;
h_pred_corr_rank(h_pred_corr_indices) = indices;

additional_roi_info_file = ['misc_glasser_files' filesep 'HCP-MMP1_UniqueRegionList.csv'];
additional_roi_info_table = readtable(additional_roi_info_file,'FileType','text','delimiter',',');
long_name = additional_roi_info_table.regionLongName;
lobe = additional_roi_info_table.Lobe;
cortex = additional_roi_info_table.cortex;
vol_mm = additional_roi_info_table.volmm;

combined_table = table(feature_mean,feature_std,h_pred_corr,h_pred_p,feature_std_rank,h_pred_corr_rank,name,long_name,cortex,lobe,x,y,z,vol_mm);
combined_table = sortrows(combined_table, 'h_pred_corr', 'descend');

disp(  nnz( (h_pred_corr_rank <= 20) & (feature_std_rank <= 20) )  )
[rho, pval] = corr(feature_std,h_pred_corr,'Type','Spearman','Tail','both');
fprintf("Spearman's rho=%g, p=%g\n", rho, pval)