% Convert .mat files to .bin binary files.
mat_file_directory = 'E:\AAL90_NoGRS_TDC_TR2_I_II_maxage_25_mat';
binary_file_directory = 'E:\aal90_short_binaries';
file_prefix = 'sub-';
file_suffix = '.mat';
directory_contents = {dir(mat_file_directory).name};
% disp( directory_contents )
is_data_file = contains( {dir(mat_file_directory).name}, file_prefix+digitsPattern+file_suffix);
% disp(is_data_file)
mat_files = directory_contents(is_data_file);
% disp(data_files)
% disp(data_files{1})
num_files = numel(mat_files);
for file_index = 1:num_files
    mat_file_name = mat_files{file_index};
    subject_id_cell = extractBetween(mat_file_name,file_prefix,file_suffix);
    subject_id_str = subject_id_cell{1};
    fprintf('subject %s\n', subject_id_str)
    mat_file_path = [mat_file_directory filesep mat_file_name];
    % disp(mat_file_path)
    data_struct = load(mat_file_path);
    % disp(data_struct)
    data_mat = data_struct.fMRI;
    disp('size')
    disp( size(data_mat) )
    % min_data = min( min(data_mat) );
    % max_data = max( max(data_mat) );
    % imshow( (data_mat - min_data)/(max_data - min_data) )
    bin_file_path = [binary_file_directory filesep 'data_ts_' subject_id_str '.bin'];
    file_id = fopen(bin_file_path, 'w');
    fwrite(file_id, data_mat, 'float64');
    fclose(file_id);
    fprintf('saved file %u of %u %s\n', file_index, num_files, bin_file_path)
end