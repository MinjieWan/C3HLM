% Clear previous variables and close all figure windows
% 清除之前的变量并关闭所有图形窗口
clear; close all

% Set the directory where the images are stored
% 设置图像存储的目录
images_dir = 'image\11_Images\';

% Get a list of all bmp images in the directory
% 获取目录中所有bmp格式图像的列表,对应其他格式图片，可自行添加
listing = cat(1, dir(fullfile(images_dir, '*.bmp')));

% Loop over each image in the directory
% 遍历目录中的每一张图像
for i_img = 1:length(listing)
    
    % Read the input image
    % 读取输入图像
    Input = imread(fullfile(images_dir,listing(i_img).name));
    
    % Extract the image name (without extension)
    % 提取图像名称（不带扩展名）
    [~, img_name, ~] = fileparts(listing(i_img).name);
    
    % Remove the '_input' part from the image name
    % 从图像名称中去除'_input'部分
    img_name = strrep(img_name, '_input', '');

    % Apply the underwater image restoration algorithm (Constrained Color Compensation and Background Light Color Space-based Haze-Line Model)
    % 应用水下图像恢复算法（基于约束色彩补偿和背景光颜色空间的雾线模型）
    output = CCCBLSHL(Input);  % Underwater Image Restoration via Constrained Color Compensation and Background Light Color Space-based Haze-Line Model
    % Create a new figure to display the original and enhanced images side by side
    % 创建一个新图形以并排显示原始图像和增强图像
    figure(1);
    
    % Display the original image in the first subplot
    % 在第一个子图中显示原始图像
    subplot(1, 2, 1);
    imshow(Input);
    title('Original Image'); % 设置标题为“原始图像”
    
    % Display the enhanced image in the second subplot
    % 在第二个子图中显示增强图像
    subplot(1, 2, 2);
    imshow(output);
    title('Enhanced Image'); % 设置标题为“增强图像”
    
    % Wait for the user to press Enter before continuing
    % 等待用户按回车键再继续
    disp('Press Enter to continue to the next image...');
    pause; % Pauses until the user presses a key
end
