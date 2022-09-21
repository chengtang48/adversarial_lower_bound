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

### TODO list (for paper)
1. Sanity check of theoretical findings on feed-forward ReLU networks
    * for random gaussian networks, understand how does the hidden dimension of the network affect the performance of k-layer linear 
    approximation performance in IBP-Lin (Prop 3.2)
        
    * demonstrate how the quality of approximation can change sharply with magnitude of network weights (Prop. 3.4) 
      
    * run IBP-Lin with increased k in k-layer approximation doesn't necessarily outperform
    the original IBP bounds (Prop. 3.5)
     

2. Additional implementation and tests
   * add support of pool layers for IBP-Lin (in linear_appx.py)
   * more tests for conv2d/pool layers for IBP-Lin
    


     

