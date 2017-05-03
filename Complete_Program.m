%% Read video into photos
clear; clc; close all;
% Select video file to extract its images
[vid_name,path] = uigetfile('C:\Users\Matthew\Documents\MATLAB\2017 Spring\Capstone\Full\Videos\4_18_17\Wet\*.mp4','MultiSelect','On');
vid = [path,'\',vid_name];
v = VideoReader(vid);
n = 0;
mkdir([path,vid_name,'_Folder']);
folder = [path,vid_name,'_Folder'];
while hasFrame(v)
    fprintf('Image: %d \n', n+1);
    imageFile = strcat('Img_', int2str(n), '.jpg');
    imageFile = [folder,'/',imageFile];
    imwrite(readFrame(v), imageFile);
    n = n + 1;
end

disp('Complete!');

%% Read Binary image

%[im_names,path] = uigetfile('C:\Users\Matthew\Documents\MATLAB\2017 Spring\Capstone\Full\Videos\4_18_17\Wet\*.jpg','MultiSelect','On');
pic0 = [folder,'\Img_0.jpg'];
img_R = imread(pic0);
img_G = rgb2gray(img_R);
img_G = imcomplement(img_G);
%img_G = imgaussfilt(img_G,3);
img_BW = imbinarize(img_G); % Get where light is
img_BW = imfill(img_BW,'holes');
imshow(pic0);
rect = getrect; % [c_min r_min width height] x is col y is row
true_pix = 0;
for r = 1:size(img_BW,1)
    for c = 1:size(img_BW,2)
        if(c>rect(1) && r>rect(2)&& c<(rect(1)+rect(3)) && r<(rect(2)+rect(4)))
            img_BW(r,c) = true;
            true_pix = true_pix+1;
        else
            img_BW(r,c) = false;
        end
    end
end
close all;
imshow(img_BW)

%cont = input('Remove more areas? (0/1): ');
cont = 0;
while(cont ==1)
    imshow(img_BW)
    rect = getrect;
    true_pix = 0;
    for r = 1:size(img_BW,1)
        for c = 1:size(img_BW,2)
            if(c>rect(1) && r>rect(2)&& c<(rect(1)+rect(3)) && r<(rect(2)+rect(4)))
                img_BW(r,c) = false;
            else
                if(img_BW(r,c) == true)
                    true_pix = true_pix+1;
                end
            end
        end
    end
    close all;
    imshow(img_BW);
    cont = input('Remove more areas? (0/1): ');
end

%save('Binary.mat','img_BW')
disp('Complete');

%% Read image files
close all;
%load Binary.mat
tic();
%[im_names,path] = uigetfile('C:\Users\Matthew\Documents\MATLAB\2017 Spring\Capstone\Full\Videos\4_18_17\Wet\*.jpg','MultiSelect','On');
if(n < 150)
   pics = n; 
else
    pics = 150;
end
im_names = cell(1,pics);
for i = 1:pics
    im_names{i} = [folder,'\Img_',num2str(i-1),'.jpg'];
end
data = zeros(size(img_BW,1),size(img_BW,2),size(im_names,2));


t = 0:size(im_names,2)-1;
for i = 1:size(im_names,2)
    img = rgb2gray(imread([im_names{i}])); % read an img
    img = im2double(img); % Convert image to double
    data(:,:,i) = img; % add image to values
end

% Constants
i_num = size(im_names,2);
i_r = size(img,1);
i_c = size(img,2);

% Take relevant image data
true_pix = count_true(img_BW);
img_data = zeros(true_pix,3,round(i_num/2));
for index = 1:i_num
    fprintf('Image: %d/%d \n',index, i_num);
    row = 1;
    for r = 1:i_r
        for c = 1:i_c
            if(img_BW(r,c) == true) 
                    img_data(row,1,index) = r; % pixel x
                    img_data(row,2,index) = c; % pixel y
                    img_data(row,3,index) = t(index); % time
                    row = row+1;
            end
        end
    end
end

x = zeros(1,index);
for i = 1:index
    x(i) = i-1;
end
% Revolutions/second
theta = 250/360;
fps = 30; % frane/sec
e = theta/fps; % Rev/frame
x = x.*e;

% Sample a pixel

% i = 10000;
% r = img_data(i,1,1);
% c = img_data(i,2,1);
% test_img = data(r,c,:);
% plot(x,test_img(:),'o');

% Optimization Problem

fact = 1;
% Minimize (i(x,y,theta) - a sin ( theta + phi))^2
phi = zeros(round(size(img_data,1)/fact),1);
temp = zeros(i_num,1);
factor = (3)*pi;
in = 1;
for i = 1:size(img_data,1) % for each pixel that is in selected region
    fprintf('Pixel: %d/%d \n',i,size(img_data,1));
    %if(mod(i,fact) == 1)
        temp = data(img_data(i,1,1),img_data(i,2,1),:); % x,y,t
        temp = temp(:);
        temp = temp-mean(temp);
        amp = max(temp);
        f1 = @(v) amp.*sin(factor*x' + v); % x'
        fmin = @(v) sum((temp-f1(v)).^2);
        phi(in,1) = fminsearch(fmin,[0]);
        %phi(i,1:2) = fsolve(fmin,[amp,0]);
        in = in+1;
    %else
        
    %end
end

time = toc()
% Plot resulting fit
% i = 200;
% temp = data(img_data(i,1,1),img_data(i,2,1),:);
% temp = temp(:);
% temp = temp-mean(temp);
% amp = max(temp);
% calc = amp*sin(factor*x'+ phi(i));
% figure(1); plot(x,temp,'bo',x,calc,'r');
% savefig([folder,'/','sine_fit_T9.fig']);
% Phi histogram

phi_h = phi(:,1);
phi_hs = sort(phi_h);
particles = 100;
figure(2);bin = histo(phi_hs,particles);
savefig([folder,'/','phase_hist_T10.fig']);

% Find peaks
p = 1:particles;
f = 25;
coef = ones(1, particles/f)/(particles/f);
avg = filter(coef, 1, bin);
[pks,loc] = findpeaks(avg,'MinPeakProminence',10);
loc = loc-2;
hold on;
plot(p,bin,'b-');
plot(p,avg,'g-');
plot(loc,pks,'ro')
hold off;

loc = loc./particles;


% Demuxing
% Need 3 images 
pix = [30,50,80];
M = zeros(3,3);
theta = [x(pix(1)),x(pix(2)),x(pix(3))];
for i = 1:3
    for j = 1:3
        M(i,j) = 1+cos(theta(i)*2)*cos(2*loc(j))+sin(2*theta(i))*sin(2*loc(j));
    end
end
M = 1/2*M;
% for every pixel
demux = zeros(i_r,i_c,3);

r1 = rect(2):rect(2)+rect(4);
c1 = rect(1):rect(1)+rect(3);

for i = r1
    for j = c1
        fprintf('Row: %d/%d\n',i,i_r);
        I_c = [data(i,j,pix(1)), data(i,j,pix(2)), data(i,j,pix(3))];
        demux(i,j,:) = (I_c*inv(M));
    end
end
disp('Complete!')


% show result
% [im_names,path] = uigetfile('C:\Users\Matthew\Documents\MATLAB\2017 Spring\Capstone\Full\Videos\4_18_17\Dry\*.jpg','MultiSelect','On');
% L1 = imread([path,im_names{1}]);
% L2 = imread([path,im_names{2}]);
% L3 = imread([path,im_names{3}]);
% % rect(1) && r>rect(2)&& c<(rect(1)+rect(3)) && r<(rect(2)+rect(4)
% c_s = rect(1); r_s = rect(2); r_e = rect(2)+rect(4); c_e = rect(1)+rect(3);

figure(); subplot(1,3,1)
imshow(demux(r1,c1,1),[]);
title('Demux light 1')
subplot(1,3,2)
imshow(demux(r1,c1,2),[]);
title('Demux light 2')
subplot(1,3,3)
imshow(demux(r1,c1,3),[]);
title('Demux light 3')

imwrite(demux,[folder,'\','demux_T10.jpg']);

time
% Real
% subplot(2,3,4)
% imshow(L1(r_s:r_e,c_s:c_e,:),[]);
% title('Real light 1')
% subplot(2,3,5)
% imshow(L2(r_s:r_e,c_s:c_e,:),[]);
% title('Real light 2')
% subplot(2,3,6)
% imshow(L3(r_s:r_e,c_s:c_e,:),[]);
% title('Real light 3')

