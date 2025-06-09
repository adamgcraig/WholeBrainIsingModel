function varargout = plot_gifti_partial(varargin)
% adapted from SPM12 @gifti/plot.m
% We made some changes so that we could use subplot. 

cdata = [];
ax = [];
if nargin == 1
    this = varargin{1};
    h = gcf;
else
    if ishandle(varargin{1})
        ax = varargin{1};
        h = figure(get(ax,'parent'));
        this = varargin{2};
    else
        this = varargin{1};
        h = gcf;
        cdata = subsref(varargin{2},struct('type','.','subs','cdata'));
    end
    % This is the part that is different:
    % We made the third argument the axes,
    % the fourth the colormap,
    % the fifth the material,
    % bumping the index into cdata to the sixth argument.
    if nargin > 2
        ax = varargin{3};
    else
        ax = [];
    end
    if nargin > 3
        cmap = varargin{4};
    else
        cmap = 'cool';
    end
    if nargin > 4
        shinyness = varargin{5};
    else
        shinyness = 'dull';
    end
    if nargin > 5
        indc = varargin{6};
    else
        indc = 1;
    end
end

% if isempty(ax), ax = axes('Parent',h); end
axis(ax,'off');
% axis(ax,'equal');
hp = patch(struct(...
    'vertices',  subsref(this,struct('type','.','subs','vertices')),...
    'faces',     subsref(this,struct('type','.','subs','faces'))),...
    'FaceColor', 'b',...
    'EdgeColor', 'none',...
    'Parent',ax);

if ~isempty(cdata)
    set(hp,'FaceVertexCData',cdata(:,indc), 'FaceColor','interp')
end

% axes(ax);
camlight;
camlight(-80,-10);
lighting(ax,'gouraud');
if strcmp(spm_check_version,'matlab')
    cameratoolbar(h);
end

material(shinyness)

% Use this version if we want NaNs to be white.
colors = [ 1.0 1.0 1.0; colormap(cmap) ];
colormap(colors)
% colormap(cmap)

if nargout
    varargout{1} = hp;
end

set(gcf,'color','black')
% set(gcf,'color','white')