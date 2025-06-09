% Make surface plots of entropy based on fMRI time series data.
% by Mianxin Liu c. 2020

clear;

trial={'rfMRI_REST1_LR','rfMRI_REST1_RL', 'rfMRI_REST2_LR', 'rfMRI_REST2_RL'};%rfMRI_REST1_LR rfMRI_REST2_LR rfMRI_REST1_RL rfMRI_REST2_RL
datapath='\\158.182.15.58\NAS_data3\mxliu\HCP_data\disk12\';
outpath='\\158.182.15.58\NAS_data3\mxliu\HCP_data\entropy_pixel\';
atlas = ft_read_cifti('\\158.182.15.58\NAS_data3\mxliu\HCP_data\atlas\MMPsurfaceAtlas\Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii');
mask=~isnan(atlas.indexmax);

fs=1/0.72;
general.fmin = 0.01; %Hz 0.2;%
general.fmax = 0.5;%5; %Hz
general.attenuation = 40;%10;%dB
general.order = 4; %4 did not work
[b, a] = cheby2(general.order,general.attenuation,[general.fmin*2/fs,general.fmax*2/fs]);%Riedner 2007

folder=dir(datapath);

% for sub=3:length(folder)
for sub=3:length(folder)
%check data
count=0;
for tr=1:4
    filepath=['\MNINonLinear\Results\' trial{tr} '\'];
    if exist([datapath folder(sub).name filepath trial{tr} '_Atlas_MSMAll_hp2000_clean.dtseries.nii'],'file')
        count=count+1;
    end
end

if count==4
Entropy=zeros(4,length(atlas.indexmax));
for tr=1:4
filepath=['\MNINonLinear\Results\' trial{tr} '\'];
data = ft_read_cifti([datapath folder(sub).name filepath trial{tr} '_Atlas_MSMAll_hp2000_clean.dtseries.nii']);

% tic;
for i=1:length(atlas.indexmax)
    if mask(i)==1
        TS=filtfilt(b,a,data.dtseries(i,:));
        Entropy(tr,i)=multiscaleDisEn(TS,1,2,3); %6 could be change to 3
    end
end
% toc;
end
save([outpath folder(sub).name '.mat'],'Entropy');
end

end

% visualize
atlas_L = gifti('\\158.182.15.58\NAS_data3\xyliu\Glasser_atlas\masks_all\L.annotation.label.gii');
atlas_R = gifti('\\158.182.15.58\NAS_data3\xyliu\Glasser_atlas\masks_all\R.annotation.label.gii');
surf_L=gifti('\\158.182.15.58\NAS_data3\mxliu\HCP_data\atlas\MMPsurfaceAtlas\Q1-Q6_RelatedParcellation210.L.midthickness_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii');
surf_R=gifti('\\158.182.15.58\NAS_data3\mxliu\HCP_data\atlas\MMPsurfaceAtlas\Q1-Q6_RelatedParcellation210.R.midthickness_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii');

for i=1:4
figure;
temp=Entropy(i,1:32492)';
% pos=find(temp~=0);
% temp(pos)=zscore(temp(pos));
show_L.cdata=temp.*mask(1:32492);
% show_L.cdata(show_L.cdata==0)=nan;
plot(surf_L,show_L);
colormap('jet');
title(['scan ' num2str(i)])
temp(temp==0)=[];
caxis([min(temp),max(temp)])
end

temp=Entropy(4,32493:end)';
show_R.cdata=temp.*mask(32493:end);
% show_L.cdata(show_L.cdata==0)=nan;
plot(surf_R,show_R);
colormap('jet');
temp(temp==0)=[];
caxis([min(temp),max(temp)])
