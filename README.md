# Endmember-Guided Nonlinear Mapping Learning for Hyperspectral Nearshore Underwater Target Detection
-----------
This project was inspired by the following two papers, I applied them to underwater target detection in remote sensing hyperspectral imagery. Due to time constraints, this work was not published.
> [1] *J. Jiao, Z. Gong and P. Zhong, Triplet Spectralwise Transformer Network for Hyperspectral Target Detection, in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-17, 2023.*
> 
> [2] *D. Hong et al., SpectralFormer: Rethinking Hyperspectral Image Classification With Transformers, in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-15, 2022.*

Hyperspectral underwater target detection is a promising and challenging task in remote sensing image processing. Existing methods face significant challenges when adapting to real nearshore environments, where cluttered backgrounds hinder the extraction of target signatures and exacerbate signal distortion.

<p align="center">
  <img src="/pics/underwater_remote_sensing_imaging.png" alt="Framework" title="underwater remote sensing imaging" width="450px">
</p>

We propose a two-stage method for underwater target detection in hyperspectral remote sensing images. In the first stage, the method utilizes a nonlinear mapping network to adaptively learn environmental parameters, aiming to replace the conventional bathymetric model. In the second stage, an improved triplet loss is employed to separate underwater target features in the image.

<p align="center">
  <img src="/pics/framework.png" alt="Framework" title="NML-UTD" width="950px">
</p>

***
### Problem Analysis
Hyperspectral underwater target detection requires identifying pixels containing the target from all pixels. However, due to the strong absorption of light by water, the spectral characteristics of the target are often 'submerged,' causing significant overlap between target samples and water background samples in most scenarios.

<p align="center">
  <img src="/pics/feature.png" alt="Framework" title="NML-UTD" width="550px">
</p>

Our proposed method demonstrates a good capability for target feature separation.

<!-- First Row -->
<p align="center">
  <img src="/pics/scene1_before_training.png" alt="Feature 1" title="River scene 1 before training" width="40%">
  <img src="/pics/scene1_after_training.png" alt="Feature 2" title="River scene 1 after training" width="40%">
</p>

<!-- Second Row -->
<p align="center">
  <img src="/pics/scene2_before_training.png" alt="Feature 1" title="River scene 2 before training" width="40%">
  <img src="/pics/scene2_after_training.png" alt="Feature 2" title="River scene 2 after training" width="40%">
</p>

### Dataset
Due to the difficulty of deploying underwater targets and the high cost of data collection, research in this area has predominantly relied on simulated data. To advance the study of underwater target detection in real-world scenarios, we used a dataset of real underwater scenes and conducted experiments on this data. The deployed underwater target is an iron plate, and the target's prior spectral data were collected onshore.

> The River Scene data sets was captured by Headwall Nano-Hyperspec imaging sensor equipped on DJI Matrice 300 RTK unmanned aerial vehicle, and it was collected at the Qianlu Lake Reservoir in Liuyang (28◦18′40.29′′ N, 113◦21′16.23′′ E), Hunan Province, China on July 31, 2021.
> The Ningxiang data set was captured using the same equipment, and it was collected at the Meihua Reservoir, Ningxiang city (27◦ 56’59.72” N, 112◦ 8’50.45” E), Hunan Province , China on January 10, 2024.


- **Download the datasets from [*here*](https://drive.google.com/file/d/19l5nimXL4ONjB6Qhl-gJaYX4sFb3zibw/view?usp=sharing), put it under the folder <u>dataset</u>.**
  
- Dataset format: mat

- River Scene1:
242×341 pixels with 270 spectral bands

- River Scene2:
255 × 261 pixels with 270 spectral bands

- River Scene3:
137 × 178 pixels with 270 spectral bands

- Ningxiang:
The Ningxiang dataset was collected in a reservoir with high mineral content and significant sediment, which may make detection challenging.

Keys: 
- data: The hyperspectral imagery contains underwater targets
- target: The target prior spectrum collected on land
- gt: The ground truth of underwater target distribution

----

### Training  &&  Testing 

**Stage 1**: Nonlinear mapping learning 

1. Modify `AE.py`
2. Run ` python AE.py `

Training for new dataset need to generate NDWI mask (If land areas are included). 
> NDWI Water Mask (require gdal):
> You can check out the `water_mask\NDWI.py` file in another project [NUN-UTD](https://github.com/Jiaxuan2021/NUN-UTD)
> - water -- 0
> - land -- 255
> - selected bands get from envi
> - GREEN.tif: green band 549.1280 nm
> - NIR.tif: near-infrared band 941.3450 nm

**Stage 2**: Feature separation network training

1. Modify `TPL.py`, using the trained weights of the nonlinear mapping network from the first stage.
2. Run ` python TPL.py `

> The trained weights are all stored in the result_* folders.


-----------------
I am no longer conducting research in this field, this is the final research project of my master's degree. I sincerely hope that this work can be of assistance to you and contribute to the advancement of the community.

*There has been limited research in this field, and many challenges remain in applying these methods to real-world scenarios. We sincerely hope that this work contributes positively to the field, despite the theoretical and practical limitations that still exist. If you have any concerns, please do not hesitate to contact liu_jiaxuan2021@163.com.*

