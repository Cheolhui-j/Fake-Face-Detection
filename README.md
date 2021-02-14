# Fake-Face-Detection

This project is a texture-based synthetic face detection. 
By combining the gram matrix that extracts the global texture and the self-attention that extracts the local texture, 
it finds the correlation of the synthetic face image and detects the synthetic face image through it.

# Method 

+ Network Architecture 
  
  ![Network Architecture]()
  
  The network structure is a combination of a gram block and a self-attention block as above.
  
+ Gram Block
  
  Each element value of the Gram matrix means the correlation value between the feature maps that match each row and column.

  ![Gram Matrix]()

  1. Transform the matrix through vectorize.
  2. Calculate the correlation matrix through the dot product.
  3. Divide the resulted value by the total number of matrices and perform nomalization to obtain the Gram matrix.
  

+ Self-Attention Block

  Each element value of the Self-Attention matrix means the correlation value between the pixels that match each row and column.

  ![self-attention]()

  1. First, Query, Key, and Value are created from the input image through three convolution layers. 
  2. Calculate the dot product of Query and Key of the input image calculated above to create an attention score.
  3. Calculate the softmax value of the attention score.
  4. Value is multiplied by the above attention score and a feature map consisting of weighted values in the same form as the input image is presented as a result.
  5. Finally, self-attention feature maps are derived by adding the created map to the input image.
  
# How to use

# Results

# TO DO

+ modify code
+ change Nework Architecture based on [Self-Attention GAN](https://arxiv.org/pdf/1805.08318.pdf)

# Reference 

Hyeonseong Jeon and Youngoh Bang and Simon S. Woo(2020), FDFtNet: Facing Off Fake Images using Fake Detection Fine-tuning Network 

[paper](https://arxiv.org/pdf/2001.01265.pdf)       [code](https://github.com/cutz-j/FDFtNet)

Zhengzhe Liu and Xiaojuan Qi and Philip Torr(2020), Global Texture Enhancement for Fake Face Detection in the Wild 

[paper](https://arxiv.org/pdf/2002.00133.pdf)
