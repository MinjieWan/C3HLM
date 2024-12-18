% Grayscale World White Balance and Histogram Auto-Contrast Adjustment
% 灰度世界白平衡和直方图自动对比度调整
function output = Grayscale_remapping(image)

% Extract the RGB channels from the input image
% 从输入图像中提取RGB通道
r = image(:, :, 1);
g = image(:, :, 2);
b = image(:, :, 3);

% Calculate the mean of each RGB channel
% 计算每个RGB通道的均值
Mean(1) = mean(mean(r));
Mean(2) = mean(mean(g));
Mean(3) = mean(mean(b));

% Find the maximum values of each RGB channel
% 找到每个RGB通道的最大值
V(1) = max(max(r));
V(2) = max(max(g));
V(3) = max(max(b));

% Find the overall maximum value and the average mean
% 找到总的最大值和平均均值
max_1 = max(V);

% Define the ratio for the saturation level (can be adjusted for more/less contrast)
% 定义饱和度级别的比例（可以调整以获得更多或更少的对比度）
ratio = [1, 1, 1];

% Set the saturation level for quantile adjustment
% 设置量化调整的饱和度级别
satLevel = 0.001 * ratio;

% Get the size of the image
% 获取图像的大小
[m, n, p] = size(image);
imgRGB_orig = zeros(p, m * n);

% Reshape each RGB channel into a row vector for processing
% 将每个RGB通道重塑为行向量以进行处理
for i = 1 : p
   imgRGB_orig(i, : ) = reshape(double(image(:, :, i)), [1, m * n]);
end

imRGB = zeros(size(imgRGB_orig));

% Histogram contrast adjustment
% 直方图对比度调整
for ch = 1 : p
    % Define quantile range for contrast adjustment
    % 定义对比度调整的分位数范围
    q = [satLevel(ch), 1 - satLevel(ch)];
    tiles = quantile(imgRGB_orig(ch, :), q); % Calculate quantiles
    temp = imgRGB_orig(ch, :);
    
    % Clip pixel values based on the quantiles
    % 根据分位数裁剪像素值
    temp(temp < tiles(1)) = tiles(1);
    temp(temp > tiles(2)) = tiles(2);
    
    % Store adjusted values
    % 存储调整后的值
    imRGB(ch, :) = temp;
    
    % Normalize the pixel values to the range [0, max_1]
    % 将像素值归一化到[0, max_1]范围
    pmin = min(imRGB(ch, :));
    pmax = max(imRGB(ch, :));
    imRGB(ch, :) = (imRGB(ch, :) - pmin) / (pmax - pmin) * max_1;
end

% Initialize the output image
% 初始化输出图像
output = zeros(size(image));

% Reshape and assign the adjusted RGB values back to the output image
% 重塑并将调整后的RGB值分配回输出图像
for i = 1 : p
        output(:, :, i) = reshape(imRGB(i, :), [m, n]); 
end

end
