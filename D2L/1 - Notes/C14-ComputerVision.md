## Chapter 14 : Computer Vision
- How to compute the reception field : from top to bottom. $r_i = (r_{i+1}-1) * s_i + k_i$.
- Image Augmantation
  - torchvision.transforms.RandomHorizontalFlip, RandomResizedCrop, ColorJitter... and use Compose to combine them. Use ToTensor to convert to tensor.
- Fine tunning
  - <figure style="text-align: center;">
      <img alt="Fine tunning" src="https://d2l.ai/_images/finetune.svg" style="background-color: white; display: inline-block;">
      <figcaption> Fine tunning </figcaption>
    </figure>
  - Change the output layer, random init it, then use a small lr to train it (Output layer can have bigger lr than other layers).
  - torchvision.datasets.ImageFolder get image dataset
- Anchor Box :
  - from a pixel get (s1, r_i(up to m)) and (s_i(up to n), r1), s is scale, r is width / height, so we have wh(n+m-1) anchor boxes, w and h is the img size. Anchor width and height is $hs\sqrt r$ and $hs / \sqrt r$. Here we assume that h == w, if it is not the case then $hs\sqrt r$ and $ws/\sqrt r$.
  - IoU, Inter over Union.
  - How to assign ground truth to anchor box? Given anchors $A_1, \ldots, A_{n_c}$, and ground truth $B_1, \ldots, B_{n_t}$. We have a matrix $X \in \mathbb R^{n_c \times n_t}$, $X_{i, j}$ is IoU of $A_i$ and $B_j$. Find max number in $\max X = X_{i_1, j_1}$, assign $B_{j_1}$ to $A_{i_1}$, abandon $i_1$ row and $j_1$ col. Find again and again until we finish assign B. For left A, find in it's row max B, if the IoU is bigger than some threshold, assign B to A.
  - We then compute the offset with $\left( \frac{\frac{x_{b}-x_{a}} {w_{a}}-\mu_{x}} {\sigma_{x}}, \frac{\frac{y_{b}-y_{a}} {h_{a}}-\mu_{y}} {\sigma_{y}}, \frac{\operatorname{l o g} \frac{w_{b}} {w_{a}}-\mu_{w}} {\sigma_{w}}, \frac{\operatorname{l o g} \frac{h_{b}} {h_{a}}-\mu_{h}} {\sigma_{h}} \right) $, mean set to 0, and $\sigma_{y} = \sigma_{x} = 0.1, \sigma_{w} = \sigma_{h} = 0.2$. Normalization will make the training more stable. And we do not compute offset for IoU < threshold anchors, so we use a mask to filter it. For other anchors, we have predict_offset that will compare with offsets for tranining.
  - non-maximum suppression，NMS: we sort all the anchors with it's class predict accuracy. In a loop, choose max one, delete all the other anchors with IoU (with the choosen anchor) larger than a threshold. We can also first erase all the anchors based on some threshold to decrease computational cost.
- Single Shot Multibox Detection
  - <figure style="text-align: center;">
      <img alt="SSD" src="https://d2l.ai/_images/ssd.svg" style="background-color: white; display: inline-block;">
      <figcaption> SSD </figcaption>
    </figure>
  - Use Conv2d rather than FC to get prediction. Flatten different feature map on the dim != 0 (batch_size), then cat them so we have connect different feature map. nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1) for class data and nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1) for offset.
- Region-based CNNs (R-CNNs)
  - RCNN:
    1. Use a select algorithm to select region proposals from the input image and resize it (to match the cnn below)
    2. pass all the regions to a pre trained CNN to get the feature (it is slow!)
    3. Use these feature train n_class + 1 SVM
    4. combine the feature and the box as a sample, train a linear regression model
    5. <figure style="text-align: center;">
          <img alt="RCNN" src="https://d2l.ai/_images/r-cnn.svg" style="background-color: white; display: inline-block;">
          <figcaption> RCNN </figcaption>
        </figure>
  - Fast RCNN
    1. still use select algorithm, but just go through CNN once, and project the proposal to the feature map.
    2. use ROI Pooling to convert these feature maps to the same output size (7 * 7), roi pool is max pool. ROI pool will first divide the region into 7 * 7 regions and then quantize, because the region of interest in the feature map has corrdinate in float type, but what we need is int type. So quantize them to int, and do the max pool, we get 7 * 7 output. There are 2 quantization, first the feature map corrdinates and the division.
    3. then use FC to get two branch : one for N(n_class + 1), and 4N for the box.
    4. <figure style="text-align: center;">
          <img alt="RCNN" src="https://d2l.ai/_images/fast-rcnn.svg" style="background-color: white; display: inline-block;">
          <figcaption> Fast RCNN </figcaption>
        </figure>
  - Faster RCNN
    1. use a RPN net (region proposal network) : use a pad = 1, k = 3 conv to extract new feature from the origin feature from the base net. Then get anchors from this new feature map, get class predict (! this is not the full class predict, it is only binary class predict -- background or not) and box predict, go through NMS and fuse with origin feature.
    2. <figure style="text-align: center;">
          <img alt="Faster RCNN" src="https://d2l.ai/_images/faster-rcnn.svg" style="background-color: white; display: inline-block;">
          <figcaption> Faster RCNN </figcaption>
        </figure>
  - Masked RCNN
    1. If we have detailed pixel information of the target, we can replace ROI pool with region of interest alignment, use bipolar interpolation to preserve spatial information. This can then use a FC to get class and box, or use a conv layer to get pixel detailed information of the origiin img (segment).
    2. ROI Align: compare with ROI pool, we do not quantize, instead we use samples. We divede the region in to the output shape we want (in float type), and for each cell do another division. For example, if we set hyperparameter sample size to $n \times n$, then divede the cell to $n \times n$. For each cell in the cell, calculate the value based on bipolar interpolation: $$F(x,y) = \sum_{a,b} F(\lfloor x \rfloor + a, \lfloor y \rfloor + b) \cdot \max(0, 1-|x-(\lfloor x \rfloor + a)|) \cdot \max(0, 1-|y-(\lfloor y \rfloor + b)|)$$ for $a, b \in \{0, 1\}$. Finally, get the average on $n \times n$ cell in cell, get the cell value.
    3. If you interested in more, we have a precise ROI pool: same as ROI align we do not quantize, but we do not do the divide into samples too. Instead we do a integral on the cell.
    4. <figure style="text-align: center;">
          <img alt="Masked RCNN" src="https://d2l.ai/_images/mask-rcnn.svg" style="background-color: white; display: inline-block;">
          <figcaption> Masked RCNN </figcaption>
        </figure>
- semantic segmentation
  - image segmentation do not care about the semantic, semantic segmentation cares (it will distinguish dog and cat), instance segmentation also called simultaneous detection and segmentation not only has semantic information but also distinguish different object (two dogs and three cats are different).
  - semantic data augmentation will not scale, only crop
- Transposed Convolution
  - Conv animation : [https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md](conv_animation)
  - Normal convolution will downsample the height and width, transposed one will increase it, it is the gradient of the std convolution. (but tconv(conv(x)) != x, there is imformation loss)
  - Y[i: i + h, j: j + w] += X[i, j] * K and the torch function is nn.ConvTranspose2d
  - Just see the code and it's comment below.
  - <figure style="text-align: center;">
      <img alt="Transposed Convolution" src="https://d2l.ai/_images/trans_conv.svg" style="background-color: white; display: inline-block;">
      <figcaption> Transposed Convolution </figcaption>
    </figure>
- Fully Convolutional Networks
  - In order to give every pixel it's class predict, we can use transposed convolution to transform the feature back into the original shape.
  - We use a pretrained ResNet base net, get rid of last two layers (an adaptive avg pool and a linear), connect it to 1 * 1 conv kernel with channel == num_classes, and a transposed convolution layer.
  - We often use bi-linear interpolation to initialize the transposed convolution layer. See in test.ipynb.
- Neural Style Transfer
  - We need two inputs, one content image, one style image. We use a pretrained CNN base model as the feature extractor, some layers will be use as content image feature, some as style image feature. The model param is frozen!!! The only parameter will change is the synthesized image. We have three loss, the total variation loss is to reduce the noise.
  - In the example in book, we use deeper layer as the content layer (do not focus on detail), and different depth layer for style layer (focus both detail and global).
  - Loss: for content loss we use torch.square(Y_hat - Y.detach()).mean(), stop the gradient flow throgh the target content image Y, it is a stated value, not a variable (and Y_hat is content_layer(content_x)). For style loss, we use the gram matrix to get correlation of the style features of different channels, for a [n, c, h, w] feature, reshape it to [c, nhw] matrix X and calculate G = XX^T, loss = torch.square(gram(Y_hat) - gram_Y.detach()).mean(). For total variation loss, $\sum_{i, j}|x_{i,j} - x_{i+1,j}|+|x_{i,j} - x_{i,j+1}|$, this will reduce the noise. Add all the content loss and style loss in all the correspoding layers with the weight and the total loss with it's weight we get the total loss. Note in book example, X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y) the first X (target data) is a copy of second X (origin content img), it is two different data.
  - <figure style="text-align: center;">
      <img alt="neural-style" src="https://d2l.ai/_images/neural-style.svg" style="background-color: white; display: inline-block;">
      <figcaption> Neural Style </figcaption>
    </figure>
<!-- <img alt="ResNeXt Block" src="https://d2l.ai/_images/rnn.svg" style="background-color: white; display: inline-block;"> -->
<!-- <img alt="ResNeXt Block" src="https://d2l.ai/_images/rnn-bptt.svg" style="background-color: white; display: inline-block;"> -->