# Probability based boundary detection
A probability based boundary(or edge) detection algorithm is implemented. Unlike the the classical edge detection algorithms like Canny and Sobel which look for only the intensity discontinuities, the probability of boundary (pb) detection algorithm considers the texture and color discontinuties in addition. This gives the algorithm a better performance compared to the baseline (Canny and Sobel) algorithms. Check the full problem statement [here](https://rbe549.github.io/spring2024/hw/hw0/#sub) for additional details. 

## Input image
<img src="Phase1/BSDS500/Images/10.jpg" width="300" height="300"/>

## Filter banks
Filter banks contain a list of filters that are applied to the input image to extract various features in it. In this project three filter banks are implemented: DoG filters, Leung-Malik filters and Gabor filters. These help us in measuring and aggregating the regional texture and brightness properties.
### Oriented Derivative of Gaussian Filters
<img src="Phase1/results/filter_banks/DoG.png" width="600" height="300"/>

### Leung-Malik Filters:
<img src="Phase1/results/filter_banks/LMS.png" width="600" height="300"/>

### Gabor Filters:
<img src="Phase1/results/filter_banks/gabor-1.png" width="600" height="300"/>
 
## Texton map
After applying all the filters shown above, we have a list of filter responses. For each pixel in the image we have a vector of filter responses that encodes the texture properties in that region. We cluster these vectors corresponding to each pixel to group together the pixels with similar texture properties. In this case we use K-mean algorithm(with K=64) to get the final texton map.

<img src="Phase1/results/t_map/10.png" width="300" height="300"/>

## Brightness map
Similarly the image was converted to gray scale and clustered the pixels with similar brightness values to generate brightness map.

<img src="Phase1/results/b_map/10.png" width="300" height="300"/>

## Color map
The image has three color channels at each pixel location. We cluster the pixels with similar color property together to generate the color map.

<img src="Phase1/results/c_map/10.png" width="300" height="300"/>

## Gradients
To calculate the oriented gradients of each map we need to get the difference between pixels at different directions and sizes(or scales). We can do these calulations efficiently by convolving the half disc masks shown below with the map. By using this approach we calculate the chi-square distances for each map.
### Half-disc Masks
<img src="Phase1/results/half_disc_mask/hd_masks1.png" width="600" height="300"/>

### Texture Gradient
<img src="Phase1/results/T_g/10.png" width="300" height="300"/>

### Brightness Gradient
<img src="Phase1/results/B_g/10.png" width="300" height="300"/>

### Color Gradient 
<img src="Phase1/results/C_g/10.png" width="300" height="300"/>

## Baselines
Output of the Canny and Sobel detectors for the image.
### Canny baseline
<img src="Phase1/BSDS500/CannyBaseline/10.png" width="300" height="300"/>

### Sobel baseline
<img src="Phase1/BSDS500/SobelBaseline/10.png" width="300" height="300"/>

## Final Pb-lite output
Final output obtained by combining the weighted average of outputs from Canny and Sobel operators with the mean gradient of the texture, brightness, and color maps.

<img src="Phase1/results/pb_lite_output/10.png" width="300" height="300"/>

## Steps to run the code
- Navigate to the `Phase1` folder

- Run the following command in the folder
```python3 Code/Wrapper.py```

## References
Arbelaez, Pablo, et al. "Contour detection and hierarchical image segmentation." IEEE transactions on pattern analysis and machine intelligence 33.5 (2010): 898-916.
