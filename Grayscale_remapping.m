% Grayscale World White Balance and Histogram Auto-Contrast Adjustment
% �Ҷ������ƽ���ֱ��ͼ�Զ��Աȶȵ���
function output = Grayscale_remapping(image)

% Extract the RGB channels from the input image
% ������ͼ������ȡRGBͨ��
r = image(:, :, 1);
g = image(:, :, 2);
b = image(:, :, 3);

% Calculate the mean of each RGB channel
% ����ÿ��RGBͨ���ľ�ֵ
Mean(1) = mean(mean(r));
Mean(2) = mean(mean(g));
Mean(3) = mean(mean(b));

% Find the maximum values of each RGB channel
% �ҵ�ÿ��RGBͨ�������ֵ
V(1) = max(max(r));
V(2) = max(max(g));
V(3) = max(max(b));

% Find the overall maximum value and the average mean
% �ҵ��ܵ����ֵ��ƽ����ֵ
max_1 = max(V);

% Define the ratio for the saturation level (can be adjusted for more/less contrast)
% ���履�Ͷȼ���ı��������Ե����Ի�ø������ٵĶԱȶȣ�
ratio = [1, 1, 1];

% Set the saturation level for quantile adjustment
% �������������ı��Ͷȼ���
satLevel = 0.001 * ratio;

% Get the size of the image
% ��ȡͼ��Ĵ�С
[m, n, p] = size(image);
imgRGB_orig = zeros(p, m * n);

% Reshape each RGB channel into a row vector for processing
% ��ÿ��RGBͨ������Ϊ�������Խ��д���
for i = 1 : p
   imgRGB_orig(i, : ) = reshape(double(image(:, :, i)), [1, m * n]);
end

imRGB = zeros(size(imgRGB_orig));

% Histogram contrast adjustment
% ֱ��ͼ�Աȶȵ���
for ch = 1 : p
    % Define quantile range for contrast adjustment
    % ����Աȶȵ����ķ�λ����Χ
    q = [satLevel(ch), 1 - satLevel(ch)];
    tiles = quantile(imgRGB_orig(ch, :), q); % Calculate quantiles
    temp = imgRGB_orig(ch, :);
    
    % Clip pixel values based on the quantiles
    % ���ݷ�λ���ü�����ֵ
    temp(temp < tiles(1)) = tiles(1);
    temp(temp > tiles(2)) = tiles(2);
    
    % Store adjusted values
    % �洢�������ֵ
    imRGB(ch, :) = temp;
    
    % Normalize the pixel values to the range [0, max_1]
    % ������ֵ��һ����[0, max_1]��Χ
    pmin = min(imRGB(ch, :));
    pmax = max(imRGB(ch, :));
    imRGB(ch, :) = (imRGB(ch, :) - pmin) / (pmax - pmin) * max_1;
end

% Initialize the output image
% ��ʼ�����ͼ��
output = zeros(size(image));

% Reshape and assign the adjusted RGB values back to the output image
% ���ܲ����������RGBֵ��������ͼ��
for i = 1 : p
        output(:, :, i) = reshape(imRGB(i, :), [m, n]); 
end

end
