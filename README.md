# Underwater Image Restoration via Constrained Color Compensation and Background Light Color Space-based Haze-Line Model
This repository contains the codes for our published paper "Underwater Image Restoration via Constrained Color Compensation and Background Light Color Space-based Haze-Line Model". 
## Abstract
The quality of underwater imaging is significantly degraded by light scattering and absorption due to water body and suspended particles. To address the issues of color distortion and contrast degradation, we propose a novel underwater image restoration method based on constrained color compensation and background light color space-based haze-line model. The method begins by applying the constrained color compensation approach to provide targeted intensity adjustments for attenuated color channels. This process corrects color distortion while simultaneously mitigating the risks of over-compensation and insufficient color saturation. Subsequently, the haze-line model is employed in the transformed background light color space to restore the underwater image. Specifically, the color corrected underwater image is transformed to a novel color space, where the background light intensity serves as the origin. In this color space, all haze-lines are identified by grouping pixels with similar color characteristics through the superpixel clustering method. Then, the transmission distribution can be estimated based on the haze-line model. Finally, the scattered light components are removed by applying the underwater de-scattering model with the estimated transmission distribution to the luminance channel of the color corrected underwater image. Comparative experiments conducted on the UIEB and UCCS underwater image datasets demonstrate the superiority of the proposed method in terms of color correction and contrast enhancement when compared with state-of-the-art underwater image restoration techniques.

![Flowchart](https://github.com/user-attachments/assets/5b367444-b0ee-451a-925c-7fa2f49437c5)


## Requirements
Matlab 2023b.

## Usage
- Run main.m to testing the performance.

## Citation
If you use our code and dataset for research, please cite our paper:

Wang J, Wan M, Xu Y, et al. Underwater Image Restoration via Constrained Color Compensation and Background Light Color Space-based Haze-Line Model[J], 2024, 62: 1-15.
