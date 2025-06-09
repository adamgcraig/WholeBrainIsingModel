function [] = save_rotated(fig_handle, plot_handle, fig_directory, fig_name)
%SAVE_ROTATED Save PNGs of the figure rotated to six different angles.
%   fig_handle_l is the handle returned by figure.
%   plot_handle_l is the handle returned by plot.
%   fig_directory is the directory to which to save the figure.
%   fig_name is the first part of the file name to use for the figure.
%   We append _[direction].png to fig_name,
%   where [direction] is top, bottom, front, back, left, or right.

save_png = @(fig_direction) saveas( fig_handle, [fig_directory filesep sprintf('%s_%s.png',fig_name,fig_direction)], 'png' );
save_png('top');
rotate(plot_handle, [0,1,0], 180);
save_png('bottom');
rotate(plot_handle, [1,0,0], 90);
save_png('front');
rotate(plot_handle, [0,1,0], 180);
save_png('back');
rotate(plot_handle, [0,1,0], 90);
save_png('left');
rotate(plot_handle, [0,1,0], 180);
save_png('right');

end