# PCA-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD

## Name: Pavan Kishore.M
## Reg No: 212221230076
## Date: 21/05/2024

## AIM:

The aim of this project is to demonstrate how to convert an image to grayscale using CUDA programming.

## Procedure:
1. Load the input image using the stb_image library.
2. Allocate memory on the GPU for the input and output image buffers.
3. Copy the input image data from the CPU to the GPU.
4. Define a CUDA kernel function that performs the grayscale conversion on each pixel of the image.
5. Launch the CUDA kernel with appropriate grid and block dimensions.
6. Copy the resulting grayscale image data from the GPU back to the CPU.
7. Save the grayscale image using the stb_image_write library.
8. Clean up allocated memory.

## Program:
```python
import cv2
from numba import cuda
import sys
from google.colab.patches import cv2_imshow

# Load the image
image = cv2.imread('jinsakai.jpg')
cv2_imshow(image)

# Check if image loading was successful
if image is None:
    print("Error: Unable to load the input image.")
    sys.exit()

# Convert the image to grayscale using CUDA
@cuda.jit
def gpu_rgb_to_gray(input_image, output_image):
    # Calculate the thread's absolute position within the grid
    x, y = cuda.grid(2)
    if x < input_image.shape[0] and y < input_image.shape[1]:
        # Convert RGB to grayscale (simple average)
        gray_value = (input_image[x, y, 0] + input_image[x, y, 1] + input_image[x, y, 2]) / 3
        output_image[x, y] = gray_value

# Allocate GPU memory for the input and output images
d_input = cuda.to_device(image)
d_output = cuda.device_array((image.shape[0], image.shape[1]), dtype=image.dtype)

# Configure the CUDA kernel
threads_per_block = (16, 16)
blocks_per_grid_x = (image.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (image.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Launch the CUDA kernel
gpu_rgb_to_gray[blocks_per_grid, threads_per_block](d_input, d_output)

# Copy the grayscale image back to the host memory
grayscale_image = d_output.copy_to_host()

# Display or save the grayscale image
cv2_imshow(grayscale_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## OUTPUT:

### Input Image

![](o1.png)


### Grayscale Image

![](o2.png)


## Result:
Thus, The CUDA program successfully converts the input image to grayscale using the GPU. The resulting grayscale image is saved as an output file. This example demonstrates the power of GPU parallelism in accelerating image processing tasks.

