function [x_trans, y_trans, z_trans] = move_hemisphere(patch_handle)
%MOVE_HEMISPHERE Translate a patch object.
%   adapted from
%   https://ww2.mathworks.cn/matlabcentral/answers/280629-transform-a-patch-using-a-transform-object

hgt = hgtransform;
M = get(hgt, 'Matrix');
xd = get(patch_handle, 'XData');
xd = xd(:);
yd = get(patch_handle, 'YData');
zd = get(patch_handle, 'ZData');
XYZ1 = [xd, yd(:), zd(:), ones(size(xd))];
XYZ1_trans = XYZ1 * M';
x_trans = XYZ1_trans(:,1);
y_trans = XYZ1_trans(:,2);
z_trans = XYZ1_trans(:,3);

end