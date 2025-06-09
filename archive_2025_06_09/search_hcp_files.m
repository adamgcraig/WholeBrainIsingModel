% Check how many files of each kind we have in the HCP data.
hcp_root = 'D:\HCP_1200\HCP_1200\';
subject_folders = {dir(hcp_root).name};
subject_folders = subject_folders(3:end);
file_suffixes = {'.nii.gz', '_Atlas_MSMAll.dtseries.nii', '_Atlas.dtseries.nii', '_Atlas_hp2000_clean_bias.dscalar.nii', '_Atlas_hp2000_clean_vn.dscalar.nii'};
scans = {'1_LR', '1_RL', '2_LR', '2_RL'};
num_subjects = numel(subject_folders);
fprintf('We have subfolders for %u subjects.\n', num_subjects)
for suffix_index = 1:numel(file_suffixes)
    file_suffix = file_suffixes{suffix_index};
    count = 0;
    for scan_index = 1:numel(scans)
        scan = scans{scan_index};
        for subject_index = 1:num_subjects
            file_name = [hcp_root, subject_folders{subject_index}, '\MNINonLinear\Results\rfMRI_REST', scan, '\rfMRI_REST', scan, file_suffix];
            if exist(file_name, 'file')
                if count == 0
                    disp(file_name)
                end
                count = count + 1;
            end
        end
    end
    fprintf('We have %u files (%g per subject) ending in %s.\n', count, count/num_subjects, file_suffix)
end
