## Modules

### Original IBP method
Implementation and tests for original IBP methods (in matrix form), 
with support of general linear layers (simple linear, conv2d, mean/max pooling),
and with support of batch processing
(equivalent to IBP-Lin with null schedule vector; see Algo 1 in paper)
#### Related files
* fast_appx.py
* utils.py

### IBP-Lin method
Implementation and tests for IBP-Lin (see Algorithm 1 in paper for pseudo code)
#### Related files
* linear_appx.py


### Models
Architecture, training, and loading functions of neural nets used for experiments
#### Related files
* relu_networks.py (small ReLU nets)
* medium_size_networks.py (LeNet-5)
* pretrained_imagenets.py (VGG imagenets)


## Experiments
1. Tested IBP with small, medium, and pretrained imagenet models
2. Tested IBP-Lin with small model

### TODO list
1. Sanity check of theoretical findings 
   * run IBP-Lin with increased k in k-layer approximation doesn't necessarily outperform
    the original IBP bounds (DONE)
     
   * How does the hidden dimension of the network affect the performance of k-layer linear 
    approximation performance in IBP-Lin
     
2. Additional implementation
   * add support of pool layers for IBP-Lin (in linear_appx.py)
   * more tests for conv2d/pool layers for IBP-Lin
    
3. Benchmarking experiments
   * Compare IBP-Lin against CROWN, CROWN-IBP, CAP, Fast-Lin on small (relu), 
     medium (LeNet-5), and large models (VGG imagenets)