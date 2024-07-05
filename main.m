clear; close all

images_dir='image\11_Images';

listing = cat(1, dir(fullfile(images_dir, '*.bmp')));

for i_img = 1:length(listing)
    Input = imread(fullfile(images_dir,listing(i_img).name));
    [~, img_name, ~] = fileparts(listing(i_img).name);
    img_name = strrep(img_name, '_input', '');

    output=CCCBLSHL(Input);%Underwater Image Restoration via Constrained Color Compensation and Background Light Color Space-based Haze-Line Model

    figure(1)
    imshow(output);
end