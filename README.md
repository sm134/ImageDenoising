
# Project Title
Zero-Reference Deep Curve Estimation (Zero-DCE) is an approach for enhancing low-light images without needing reference images. It formulates the enhancement task as estimating an image-specific tonal curve using a deep neural network called DCE-Net.

## Key Concepts
Image-Specific Tonal Curve:
Zero-DCE uses DCE-Net to estimate high-order tonal curves for each pixel in a low-light image.
These curves adjust the dynamic range of the image, enhancing its brightness and contrast.

Input and Output:
The network takes a low-light image as input.
It outputs high-order tonal curves that are applied to each pixel, resulting in an enhanced image.

Dynamic Range Adjustment:
The adjustment is done in a pixel-wise manner to ensure that the enhanced image retains its range and the contrast between neighboring pixels.

Inspiration from Photo Editing:
The curve estimation mimics the curves adjustment feature in photo editing software like Adobe Photoshop, where users can tweak the tonal range of an image.

Non-Reference Training:
Unlike traditional methods that require pairs of low-light and well-lit images for training, Zero-DCE uses non-reference loss functions.
These loss functions indirectly measure the quality of the enhancement and guide the network training without needing reference images.
## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/sm134/ImageDenoising.git 
    ```
    
2. Navigate to the project directory:
    ```bash
    cd ImageDenoising
    ```
    
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

    
## Training

To train the model, run:
```bash
python main.py
```

## Research Paper
This is an implementation of research paper :https://arxiv.org/pdf/2001.06826
