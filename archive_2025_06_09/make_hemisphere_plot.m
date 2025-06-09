function [fig_handle, plot_handle, light_handle] = make_hemisphere_plot(feature_expanded, surface_directory, title_text, color_bounds, side)
%UNTITLED Plot either the left (side='L') or right (side='R') hemisphere.
%   feature_expanded a list of values to map to colors at grayordinates
%   atlas_directory the directory from which to load the suface file
%   title_text text for the title of the figure
%   color_bounds [min, max] pair of values to map to the end colors
%   Values outside of this range do not get mapped to colors.
%   side 'L' or 'R'
%   fig_handle handle of the figure we generated
%   light_handle handle of the light object we generated
%   This function requires the spm12 MATLAB library.

% atlas = gifti([atlas_directory filesep side '.annotation.label.gii']);
surf_hemisphere = gifti([surface_directory filesep 'Q1-Q6_RelatedParcellation210.' side '.midthickness_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii']);

num_values = numel(feature_expanded);
halfway_point = floor(num_values/2);
if side == 'L'
    hemisphere_indices = 1:halfway_point;
    side_title = 'left';
else
    hemisphere_indices = (halfway_point+1):num_values;
    side_title = 'right';
end

fig_handle = figure;
show_hemisphere.cdata = feature_expanded(hemisphere_indices);
plot_handle = plot(surf_hemisphere, show_hemisphere);
colormap('cool');
clim(color_bounds)
material dull
light_handle = camlight;

colorbar
title([title_text ' (' side_title ' hemisphere)'])

end