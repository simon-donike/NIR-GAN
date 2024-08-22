# NIR-SRGAN: Synthetic NIR band from RGB Satellite Imagery
![Sample Result](resources/banner.png)



## Overview

**NIR_SRGAN** is a repository dedicated to synthesizing Near-Infrared (NIR) imagery from high-resolution RGB images using a Generative Adversarial Network (GAN). This project leverages RGB images at 2.5m resolution from the National Agriculture Imagery Program (NAIP) and synthesizes them to resemble Sentinel-2 (Sen2) imagery. The core functionality involves training a GAN to generate synthetic NIR images solely from the RGB bands.

## Project Objectives

- **Resolution Enhancement**: Utilize RGB images from NAIP with a 2.5m spatial resolution to produce high-quality synthetic images that mimic the characteristics of Sentinel-2 imagery.
  
- **NIR Synthesis**: Employ a GAN architecture to generate fake NIR bands using only the RGB bands from the input images.

## Data

### Input Data

- **NAIP Imagery**: RGB images at 2.5m resolution, captured by the National Agriculture Imagery Program (NAIP).
- **SEN2 Images**: Alternatively, Sen2 images at 10m can be used.

### Output Data

- **Synthesized Sentinel-2 Imagery**: RGB images synthetically adjusted to resemble Sentinel-2 data.
- **Fake NIR Images**: Generated NIR bands based solely on the input RGB bands.

## Methodology

The workflow of this project can be broken down into the following steps:

1. **Data Preprocessing**:
   - RGB images from NAIP are collected and preprocessed to match the spatial and spectral characteristics of Sentinel-2 imagery.
   
2. **Image Synthesis**:
   - The RGB images are fed into a GAN architecture designed to enhance and adjust the RGB bands to mimic Sentinel-2 imagery.

3. **NIR Band Generation**:
   - The key innovation of this project is the generation of fake NIR bands from the adjusted RGB images using a specialized GAN model.
   
4. **Model Training**:
   - The GAN model is trained on paired data of RGB and real NIR bands to learn the mapping between the RGB bands and the corresponding NIR band.
   
5. **Evaluation**:
   - The performance of the synthetic NIR bands is evaluated using various quantitative metrics and visual inspection.

## Architecture

### Generative Adversarial Network (GAN)

The GAN architecture consists of:

- **Generator**: Learns to generate synthetic NIR bands from the input RGB images.
- **Discriminator**: Evaluates the realism of the generated NIR bands, distinguishing them from real NIR images.

### Installation

Clone the repository:

```bash
git clone https://github.com/simon-donike/NIR_SRGAN.git
cd NIR_SRGAN
