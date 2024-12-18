function [img_Restored] = CCCBLSHZ(Input,~)
% CCCBLSHZ: This function performs underwater image restoration
% using a constrained color compensation method and a haze-line model.
% This function restores the color channels by adjusting the balance
% between each channel and improving the clarity of the image.
%
% Parameters:
%   Input (3D matrix): The input image in RGB format
%
% Returns:
%   img_Restored (3D matrix): The restored image

% 读取并归一化图像数据
im_c = double(Input) / 255; % Normalize the input image to the range [0, 1]

% 获取图像尺寸
[m, n, c] = size(im_c); % Get the dimensions of the image (rows, columns, channels)

% 将图像转为灰度并保存
im_c_remap = Grayscale_remapping(im_c); % Convert the image to grayscale using the Grayscale_remapping function

% 计算每个颜色通道的总和
sum_r = sum(im_c(:,:,1), 'all'); % Sum of the red channel
sum_g = sum(im_c(:,:,2), 'all'); % Sum of the green channel
sum_b = sum(im_c(:,:,3), 'all'); % Sum of the blue channel

% 根据通道总和调整 im_c_remap 的颜色通道
% Adjust the color channel of im_c_remap based on the total sum of the channels.
if sum_r > max(sum_g, sum_b)
    % 如果红色通道总和最大，保持绿色和蓝色通道不变
    im_c_remap(:,:,2) = im_c(:,:,2); % Keep the green channel unchanged
    im_c_remap(:,:,3) = im_c(:,:,3); % Keep the blue channel unchanged
elseif sum_g > max(sum_r, sum_b)
    % 如果绿色通道总和最大，保持红色和蓝色通道不变
    im_c_remap(:,:,1) = im_c(:,:,1); % Keep the red channel unchanged
    im_c_remap(:,:,3) = im_c(:,:,3); % Keep the blue channel unchanged
else
    % 如果蓝色通道总和最大，保持红色和绿色通道不变
    im_c_remap(:,:,1) = im_c(:,:,1); % Keep the red channel unchanged
    im_c_remap(:,:,2) = im_c(:,:,2); % Keep the green channel unchanged
end

% 初始化颜色通道矩阵
% Traverse each pixel and fill the matrix
im_c_r_x = zeros(m*n, 4); % Matrix for the red channel
im_c_g_x = zeros(m*n, 4); % Matrix for the green channel
im_c_b_x = zeros(m*n, 4); % Matrix for the blue channel

% 遍历每个像素点并填充矩阵

for i = 1:m
    for j = 1:n
        idx = (i-1)*n + j; % Calculate the linear index for the current pixel

        % 红色通道
        im_c_r_x(idx, :) = [im_c_remap(i, j, 1), i, j, im_c(i, j, 1)]; % Store red channel data with pixel coordinates and original red value

        % 绿色通道
        im_c_g_x(idx, :) = [im_c_remap(i, j, 2), i, j, im_c(i, j, 2)]; % Store green channel data with pixel coordinates and original green value

        % 蓝色通道
        im_c_b_x(idx, :) = [im_c_remap(i, j, 3), i, j, im_c(i, j, 3)]; % Store blue channel data with pixel coordinates and original blue value
    end
end

% 根据第四列（原始图像的颜色值）对每个通道矩阵排序
im_c_r_x = sortrows(im_c_r_x, 4); % Sort red channel data by original red values
im_c_g_x = sortrows(im_c_g_x, 4); % Sort green channel data by original green values
im_c_b_x = sortrows(im_c_b_x, 4); % Sort blue channel data by original blue values

% 计算增益并进行颜色通道的调整
if sum_r > max(sum_g, sum_b)
    % 红色通道最强
    gain = im_c_r_x(:,1) ./ im_c_r_x(:,4); % Calculate gain for red channel
    gain(isnan(gain)) = 0; % Handle NaN values (avoid division by zero)
    a1 = max(im_c_r_x(:,1)) / max(im_c_g_x(:,4)); % Ratio of red to green
    a2 = max(im_c_r_x(:,1)) / max(im_c_b_x(:,4)); % Ratio of red to blue

    % 更新绿色和蓝色通道
    for i = 1:m*n
        im_c_remap(im_c_g_x(i,2), im_c_g_x(i,3), 2) = im_c_g_x(i,1) * gain(i) * a1; % Adjust green channel
        im_c_remap(im_c_b_x(i,2), im_c_b_x(i,3), 3) = im_c_b_x(i,1) * gain(i) * a2; % Adjust blue channel
    end
elseif sum_g > max(sum_r, sum_b)
    % 绿色通道最强
    gain = im_c_g_x(:,1) ./ im_c_g_x(:,4); % Calculate gain for green channel
    gain(isnan(gain)) = 0; % Handle NaN values
    a1 = max(im_c_g_x(:,1)) / max(im_c_r_x(:,4)); % Ratio of green to red
    a2 = max(im_c_g_x(:,1)) / max(im_c_b_x(:,4)); % Ratio of green to blue

    % 更新红色和蓝色通道
    for i = 1:m*n
        im_c_remap(im_c_r_x(i,2), im_c_r_x(i,3), 1) = im_c_r_x(i,1) * gain(i) * a1; % Adjust red channel
        im_c_remap(im_c_b_x(i,2), im_c_b_x(i,3), 3) = im_c_b_x(i,1) * gain(i) * a2; % Adjust blue channel
    end
else
    % 蓝色通道最强
    gain = im_c_b_x(:,1) ./ im_c_b_x(:,4); % Calculate gain for blue channel
    gain(isnan(gain)) = 0; % Handle NaN values
    a1 = max(im_c_b_x(:,1)) / max(im_c_r_x(:,4)); % Ratio of blue to red
    a2 = max(im_c_b_x(:,1)) / max(im_c_g_x(:,4)); % Ratio of blue to green

    % 更新红色和绿色通道
    for i = 1:m*n
        im_c_remap(im_c_r_x(i,2), im_c_r_x(i,3), 1) = im_c_r_x(i,1) * gain(i) * a1; % Adjust red channel
        im_c_remap(im_c_g_x(i,2), im_c_g_x(i,3), 2) = im_c_g_x(i,1) * gain(i) * a2; % Adjust green channel
    end
end

% 将 im_c_remap 赋值给 im_c
im_c = im_c_remap;

% 计算每个颜色通道的总和
sum_r = sum(sum(im_c(:,:,1))); % 计算红色通道的总和
sum_g = sum(sum(im_c(:,:,2))); % 计算绿色通道的总和
sum_b = sum(sum(im_c(:,:,3))); % 计算蓝色通道的总和

% 设置阈值、alpha 和 beta 参数
Thr = 10; % 阈值
alpha = 0.8; % 调整系数 alpha
beta = 0.2; % 调整系数 beta

% 根据各通道总和判断主导颜色通道
% Determine the dominant color channel based on the sum of each channel
if ((sum_g > sum_r) && (sum_g > sum_b)) % 如果绿色通道最大
    g_r = sum_g / sum_r; % 计算绿色与红色的比值
    g_b = sum_g / sum_b; % 计算绿色与蓝色的比值

    % 如果比值超过阈值，则将比值限制在阈值范围内
    if (g_r > Thr)
        g_r = Thr;
    end
    if (g_b > Thr)
        g_b = Thr;
    end

    % 调整红色通道和蓝色通道
    im_c(:,:,1) = im_c(:,:,1) * alpha + (g_r - alpha - beta) * sum_r * (im_c(:,:,2)) / sum_g + beta * sum_r / (m * n);
    im_c(:,:,2) = im_c(:,:,2); % 绿色通道保持不变
    im_c(:,:,3) = im_c(:,:,3) * alpha + (g_b - alpha - beta) * sum_b * (im_c(:,:,2)) / sum_g + beta * sum_b / (m * n);

elseif ((sum_r > sum_g) && (sum_r > sum_b)) % 如果红色通道最大
    r_g = sum_r / sum_g; % 计算红色与绿色的比值
    r_b = sum_r / sum_b; % 计算红色与蓝色的比值

    % 如果比值超过阈值，则将比值限制在阈值范围内
    if (r_g > Thr)
        r_g = Thr;
    end
    if (r_b > Thr)
        r_b = Thr;
    end

    % 调整绿色通道和蓝色通道
    im_c(:,:,1) = im_c(:,:,1); % 红色通道保持不变
    im_c(:,:,2) = im_c(:,:,2) * alpha + (r_g - alpha - beta) * sum_g * (im_c(:,:,1)) / sum_r + beta * sum_g / (m * n);
    im_c(:,:,3) = im_c(:,:,3) * alpha + (r_b - alpha - beta) * sum_b * (im_c(:,:,1)) / sum_r + beta * sum_b / (m * n);

else % 如果蓝色通道最大
    b_r = sum_b / sum_r; % 计算蓝色与红色的比值
    b_g = sum_b / sum_g; % 计算蓝色与绿色的比值

    % 如果比值超过阈值，则将比值限制在阈值范围内
    if (b_r > Thr)
        b_r = Thr;
    end
    if (b_g > Thr)
        b_g = Thr;
    end

    % 调整红色通道和绿色通道
    im_c(:,:,1) = im_c(:,:,1) * alpha + (b_r - alpha - beta) * sum_r * (im_c(:,:,3)) / sum_b + beta * sum_r / (m * n);
    im_c(:,:,2) = im_c(:,:,2) * alpha + (b_g - alpha - beta) * sum_g * (im_c(:,:,3)) / sum_b + beta * sum_g / (m * n);
    im_c(:,:,3) = im_c(:,:,3); % 蓝色通道保持不变
end

% 对调整后的图像进行归一化处理
im_c = im_c / max(max(max(im_c))); % 归一化图像，使最大值为 1
im_c = min(im_c, 1); % 将图像值限制在 [0, 1] 范围内
im_c = max(im_c, 0); % 将图像值限制在 [0, 1] 范围内

% 估计大气光并重新塑形为 1x1x3 的矩阵
A = (reshape(estimate_airlight(im_c), 1, 1, 3));

% 初始化一个与图像大小相同的矩阵 delt_I，用于存储颜色通道的变化量
delt_I = zeros(m, n, c);

% 遍历每个颜色通道并将图像的每个通道赋值给 delt_I
for ColorCh = 1:1:c
    delt_I(:,:,ColorCh) = im_c(:,:,ColorCh);  % 将 im_c 中的每个颜色通道赋值给 delt_I
end

% 将 delt_I 重新塑形为二维矩阵，其中每行表示一个像素点，列表示颜色通道
delt_I_reshape = reshape(delt_I, [m*n, c]);  % 重新塑形为 (m*n) 行，c 列的矩阵

% Reshape the delta image and append a column of ones for homogeneous coordinates
% 重塑 delta_I 并在其后附加一列1，转换为齐次坐标
delt_I_reshape_H = [delt_I_reshape, ones(m*n, 1)];

% Reshape vector A into a 3x1 column vector and calculate its magnitude
% 将向量 A 重塑为 3x1 的列向量并计算其模长
A_reshape = reshape(A, [3, 1]);
A_ra = sqrt(sum(A_reshape.^2));  % Equivalent to sqrt(A_reshape(1)^2 + A_reshape(2)^2 + A_reshape(3)^2)

% Define the rotation angles for the X and Y axes
% 定义 X、Y 和 Z 轴的旋转角度
a = 90;
b = -180;
% Rotation matrix for rotation around the X axis
% X轴旋转矩阵
Rx = [1, 0, 0;
    0, cosd(a), -sind(a);
    0, sind(a), cosd(a)];

% Rotation matrix for rotation around the Y axis
% Y轴旋转矩阵
Ry = [cosd(b), 0, sind(b);
    0, 1, 0;
    -sind(b), 0, cosd(b)];

% Combined rotation matrix for Y, X rotations and translation vector
% 组合 Y、X 旋转矩阵和平移向量
RT = [Ry * Rx, A_reshape];

% Initialize the rotated image array
% 初始化旋转后的图像数组
delt_I_Rota = zeros(m*n, 3);

% Perform the matrix multiplication to rotate the image
% 进行矩阵乘法以旋转图像
delt_I_Rota = (RT * delt_I_reshape_H')';

% Take the absolute value of the rotated image (to avoid negative values)
% 对旋转后的图像取绝对值（避免负值）
delt_I_Rota = abs(delt_I_Rota);

% Reshape the rotated image back to its original dimensions
% 将旋转后的图像重塑回原始尺寸
I_Rota = reshape(delt_I_Rota, [m, n, c]);

% Define the number of superpixels, which can be modified based on the image resolution.
% 定义超像素的数量,可以根据图像分辨率进行修改
PixelNum = 50000;

% Compute superpixels for the rotated image
% 计算旋转后的图像的超像素
[Label, ~] = superpixels(I_Rota, PixelNum);

% Reshape the label matrix to a column vector for processing
% 将标签矩阵重塑为列向量以便处理
Label_superpixel = reshape(Label, [m*n, 1]);

% Get the total number of unique superpixels
% 获取超像素的总数
n_points = max(Label_superpixel);

% Calculate the radius (magnitude) for each pixel in the rotated image
% 计算旋转图像中每个像素的半径（模长）
radius = sqrt(delt_I_Rota(:,1).^2 + delt_I_Rota(:,2).^2 + delt_I_Rota(:,3).^2);

% Compute the mean value of each color channel (Red, Green, Blue) for each superpixel
% 为每个超像素计算每个颜色通道（红色、绿色、蓝色）的均值
red_mean = accumarray(Label_superpixel, delt_I_Rota(:, 1), [n_points, 1], @mean);
green_mean = accumarray(Label_superpixel, delt_I_Rota(:, 2), [n_points, 1], @mean);
blue_mean = accumarray(Label_superpixel, delt_I_Rota(:, 3), [n_points, 1], @mean);

% Combine the mean values of red, green, and blue channels into a single matrix
% 将红色、绿色和蓝色通道的均值合并为一个矩阵
vec = [red_mean, green_mean, blue_mean];

% Reshape the radius to a column vector
% 将半径重塑为列向量
radius = reshape(radius, [m*n, 1]);

% Convert the RGB values to LAB color space using a color transformation
% 使用颜色变换将 RGB 值转换为 LAB 色彩空间
transform_lab = makecform('srgb2lab');  % Create a transformation structure for RGB to LAB
transform_lab = applycform(vec, transform_lab);  % Apply the transformation
transform_lab = lab2double(transform_lab);  % Convert the LAB values to double precision

% Extract the 'a' and 'b' channels from the LAB color space
% 从 LAB 色彩空间中提取 'a' 和 'b' 通道
ab = [transform_lab(:, 2), transform_lab(:, 3)];

% Apply k-means clustering to the 'ab' values to create 2000 clusters
% 对 'ab' 值应用 k-means 聚类，生成 2000 个簇
T = kmeans(ab, 2000);

% Find the maximum cluster index (the cutoff point)
% 查找最大簇索引（分割点）
cutoff = max(T);

% Reshape the superpixel labels and calculate max and standard deviation of radius for each cluster
% 重塑超像素标签，并为每个簇计算半径的最大值和标准差
T_superpixels = reshape(T(Label_superpixel), [m*n, 1]);
radius_max_reshape = accumarray(T_superpixels, radius(:), [cutoff, 1], @max);
radius_std_reshape = accumarray(T_superpixels, radius(:), [cutoff, 1], @std);

% Reshape the radius and max/std values back to the image dimensions
% 将半径、最大值和标准差重新塑造成图像维度
radius_reshape = reshape(radius, [m, n, 1]);
radius_max_reshape = reshape(radius_max_reshape(T_superpixels), [m, n, 1]);
radius_std_reshape = reshape(radius_std_reshape(T_superpixels), [m, n, 1]);

% Normalize the standard deviation values to the range [0, 1]
% 将标准差值归一化到 [0, 1] 范围
radius_std_weight = (radius_std_reshape ./ max(max(radius_std_reshape)));

% Estimate the lower bound of the transmission based on the image and A vector
% 根据图像和 A 向量估计传输的下界
trans_lower_bound = 1 - min(bsxfun(@rdivide, im_c, reshape(A, 1, 1, 3)), [], 3);

% Estimate the transmission by dividing the radius by the maximum radius for each superpixel
% 通过将半径除以每个超像素的最大半径来估计传输
transmission_estimation = radius_reshape ./ radius_max_reshape;

% Ensure the transmission is not less than the lower bound for transmission estimation
% 确保传输不小于传输估计的下界
transmission = max(transmission_estimation, trans_lower_bound);

% Regularization parameter for WLS optimization
% WLS 优化的正则化参数
lambda = 0.1;

% Perform Weighted Least Squares (WLS) optimization for transmission
% 对传输进行加权最小二乘 (WLS) 优化
transmission = wls_optimization(transmission, radius_std_weight, im_c, lambda);

% Set up color space transformations (RGB to LAB and back)
% 设置色彩空间转换（从 RGB 到 LAB 和从 LAB 到 RGB）
transform_lab = makecform('srgb2lab');
transform_lab_A = makecform('srgb2lab');
transform_rgb = makecform('lab2srgb');

% Convert the input image from RGB to LAB color space
% 将输入图像从 RGB 转换为 LAB 色彩空间
transform_lab = applycform(im_c, transform_lab);
transform_lab = lab2double(transform_lab); % Convert LAB values to double precision

% Extract the L, a, and b channels from the LAB image
% 从 LAB 图像中提取 L、a 和 b 通道
L = transform_lab(:,:,1);

% Convert the airlight vector A from RGB to LAB space
% 将空气光向量 A 从 RGB 转换为 LAB 空间
air_Lab = applycform(A, transform_lab_A);
air_Lab = lab2double(air_Lab);
air_l = air_Lab(:,:,1);

% Apply the transmission correction to the L channel
% 对 L 通道应用传输修正
transform_lab(:,:,1) = (L(:,:) - (1 - transmission) .* air_l) ./ max(transmission, 0.2);

% Normalize and adjust the L channel to improve contrast
% 归一化和调整 L 通道以提高对比度
Lab_l = zeros(m, n);
Lab_l(:,:) = transform_lab(:,:,1);
Lab_l(:,:) = (Lab_l(:,:) - min(min(Lab_l(:,:)))) / (max(max(Lab_l(:,:))) - min(min(Lab_l(:,:))));
Lab_l(:,:) = imadjust(Lab_l(:,:));
Lab_l(:,:) = Lab_l(:,:) * 100; % Scale to the range [0, 100]

% Apply edge detection filters to the adjusted L channel
% 对调整后的 L 通道应用边缘检测滤波器
H1 = [-1 -1 -1; 0 0 0; 1 1 1]; % Horizontal edge detection filter
dx = filter2(H1, Lab_l);

H2 = [-1 0 1; -1 0 1; -1 0 1]; % Vertical edge detection filter
dy = filter2(H2, Lab_l);

% Adjust the L channel by adding a weighted version of the gradient magnitude
% 通过添加加权的梯度幅值来调整 L 通道
Lab_l = Lab_l + sqrt(dx.^2 + dy.^2) * 0.1;

% Update the L channel in the LAB image with the modified values
% 使用修改后的值更新 LAB 图像中的 L 通道
transform_lab(:,:,1) = Lab_l(:,:);

% Convert the LAB image back to RGB color space
% 将 LAB 图像转换回 RGB 色彩空间
img_dehazed = applycform(transform_lab, transform_rgb);

% Apply a radiometric correction to the final dehazed image
% 对最终的去雾图像应用辐射校正
img_Restored = img_dehazed.^(1.05); % Apply gamma correction
end