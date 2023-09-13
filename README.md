# Attention
This is a collection of links, descriptions and experiments performed while doing a deep dive on attention mechanisms.

## Experiment One
The first resource I found that significantly improved my understanding of the paper [Attention Is All You Need](https://arxiv.org/pdf/1706.03762 "Attention Is All You Need") was from the write up [Transformers from Scratch in PyTorch](https://fkodom.substack.com/p/transformers-from-scratch-in-pytorch "Transformers from Scratch in PyTorch Write Up"). The author takes excerpts from "Attention Is All You Need" and implements the Scaled Dot-Product Attention and Multi-Head Attention Mechanism one piece at a time using PyTorch. There is also a [code repo](https://github.com/fkodom/transformer-from-scratch "Transformers from Scratch in PyTorch Repo") with the implementation with unit tests done in pytest. I've forked the repo here and add the file [transformer-from-scratch/test.py](https://github.com/holestine/transformer-from-scratch/blob/main/test.py) to make it easy to step through with VSCode. 

## Experiment Two
Since I come from a Computer Vision background I was interested to see how well transformers work with images and came across the [repo for vit-pytorch](https://github.com/lucidrains/vit-pytorch "vit-pytorch") which has implementations for several different attention mechanisms. I then copied the [PyTorch MNIST Sample](https://github.com/pytorch/examples/blob/main/mnist/main.py) to [mnist.py](./mnist.py) and replaced the CNN model with a single multi-headed vision transformer.

```
self.model = ViT(
    image_size  =  784,   # MNIST image size is 28*28=784
    patch_size  =    7,   # Number of patches will be image_size/patch_size^2
    num_classes =   10,   # Classify digits 0-9
    dim         = 1024,   # Length of encoding
    depth       =    6,   # Number of Transformer blocks
    heads       =   16,   # Number of heads in Multi-head Attention layer
    mlp_dim     = 2048,   # Dimension of the MLP (FeedForward) layer
    channels    =    1,   # Number of input channels
    dropout     =    0.1, # Dropout rate
    emb_dropout =    0.1  # Embedding dropout rate
    )
```

Then switching to an Adam optimizer with a learning rate of .0001 resulted in an accuracy of 99%. For reference the original CNN model trains to 99%.

## Experiment Three

I suppose the next thing to do is to create a hybrid model. So I started with the [PyTorch MNIST Sample](https://github.com/pytorch/examples/blob/main/mnist/main.py) again but this time kept the convolutional layers and replaced the fully connected layers with the following vision transformer. The values chosen for initialization are a bit lower than what is generally seen but they keep the number of parameters between the various models within the same order of magnitude. 

```
self.vit = ViT(
    image_size  = 144,
    patch_size  =   6,
    num_classes =  10,
    dim         = 128,
    depth       =   6,
    heads       =   4,
    mlp_dim     = 256,
    channels    =  64,
    dropout     =   0.1,
    emb_dropout =   0.1
)
```

Applying these same modifications to the *dim*, *depth*, *heads* and *mlp_dim* parameters for the transformer only model as well as DeepViTNet, another attention mechanism from [vit-pytorch](https://github.com/lucidrains/vit-pytorch "vit-pytorch") and incorporating the [CIFAR10 and CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html "CIFAR website") datasets yields the following results. Subsequent runs produce similar numbers.


|Model Type   | Dataset     | Max Accuracy| Average Inference Time (ms)| Parameters|
|-------------|-------------|------------:|---------------------------:|----------:|
|CNN          | MNIST       | 97.05       |  1.33                      | 1,199,882 |
|CNN          | CIFAR10     | 57.78       |  1.33                      | 1,626,442 |
|CNN          | CIFAR100    | 22.05       |  1.24                      | 1,638,052 |
|ViT          | MNIST       | 97.87       |  9.46                      | 1,614,866 |
|ViT          | CIFAR10     | 51.0        |  9.88                      | 1,812,106 |
|ViT          | CIFAR100    | 19.72       |  9.90                      | 1,823,716 |
|DeepViTNet   | MNIST       | 97.6        | 13.02                      | 1,615,010 |
|DeepViTNet   | CIFAR10     | 48.03       | 13.30                      | 1,812,250 |
|DeepViTNet   | CIFAR100    | 17.89       | 13.31                      | 1,823,860 |
|Hybrid       | MNIST       | 99.2        | 10.40                      | 1,591,652 |
|Hybrid       | CIFAR10     | 69.52       | 10.40                      | 1,727,012 |
|Hybrid       | CIFAR100    | 34.97       | 10.37                      | 1,727,012 |

The most interesting thing here is that the CNN model executes much faster than all other models and the Hybrid model produces the best results. Let's adjust the initialization of the vision transformer to more commonly seen values, in this case the values from the paper [AN IMAGE IS WORTH 16X16 WORDS](https://arxiv.org/pdf/2010.11929.pdf "PDF") for the ViT-Base model. So the initialization of the transformer only model for the MNIST dataset looks like the following and the same values for *dim*, *depth*, *heads* and *mlp_dim* are also used in the DeepViNet and Hybrid models.

```
self.model = ViT(
    image_size  = 28*28,
    patch_size  =    14,
    num_classes =    10,
    dim         =   768,
    depth       =    12,
    heads       =    12,
    mlp_dim     =  3072,
    channels    =     1,
    dropout     =     0.1,
    emb_dropout =     0.1
)
```

And these are the results we get showing the values before and after where relevant. 

|Model Type   | Dataset     | Max Accuracy   | Average Inference Time (ms)| Parameters             |
|-------------|-------------|---------------:|---------------------------:|-----------------------:|
|CNN          | MNIST       | 97.05 -> 98.70 |  1.33 -> 1.34              | 1,199,882              |
|CNN          | CIFAR10     | 57.78 -> 57.39 |  1.33 -> 1.33              | 1,626,442              |
|CNN          | CIFAR100    | 22.05 -> 22.38 |  1.24 -> 1.26              | 1,638,052              |
|ViT          | MNIST       | 97.87 -> 98.87 |  9.46 -> 13.04             | 1,614,866 -> 87,599,250|
|ViT          | CIFAR10     | 51.0  -> 58.70 |  9.88 -> 13.98             | 1,812,106 -> 88,776,970|
|ViT          | CIFAR100    | 19.72 -> 32.87 |  9.90 -> 14.24             | 1,823,716 -> 88,846,180|
|DeepViTNet   | MNIST       | 97.6  -> 98.54 | 13.02 -> 16.20             | 1,615,010 -> 87,601,266|
|DeepViTNet   | CIFAR10     | 48.03 -> 55.71 | 13.30 -> 19.65             | 1,812,250 -> 88,778,986|
|DeepViTNet   | CIFAR100    | 17.89 -> 29.32 | 13.31 -> 19.88             | 1,823,860 -> 88,848,196|
|Hybrid       | MNIST       | 99.2  -> 99.37 | 10.40 -> 12.57             | 1,591,652 -> 87,275,146|
|Hybrid       | CIFAR10     | 69.52 -> 73.02 | 10.40 -> 14.86             | 1,727,012 -> 88,076,106|
|Hybrid       | CIFAR100    | 34.97 -> 45.87 | 10.37 -> 15.21             | 1,727,012 -> 88,145,316|

The CNN model was unmodified and the numbers produced correlate with the previous run. For all other models that contain attention mechanisms we can see that the number of parameters has increased significantly however the increase in inference time is more modest indicating that many of the additional operations are done in parallel. The Hybrid model once again produces the best results and has an accuracy score twice that of the CNN model for the CIFAR100 dataset. This seems to indicate that as the complexity of the task increases that the benefit we get from attention mechanisms becomes more pronounced. In order to get a sense of the effect the depth parameter has in the vision transformer I ran this same test again but initialized the value to 6 as it was before (instead of 12 from the last experiment). The following table shows a third column where relevenat and that the accuracy was not significantly affected, the inference time was brought back down and the parameter count is significantly reduced yet still much higher than the CNN only model. You can run this experiment for yourself using the file [experiments.py](./experiments.py) with the correct parameters.

|Model Type   | Dataset     | Max Accuracy            | Average Inference Time (ms)| Parameters                           |
|-------------|-------------|------------------------:|---------------------------:|-------------------------------------:|
|CNN          | MNIST       | 97.05 -> 98.70          |  1.33 -> 1.34              | 1,199,882                            |
|CNN          | CIFAR10     | 57.78 -> 57.39          |  1.33 -> 1.33              | 1,626,442                            |
|CNN          | CIFAR100    | 22.05 -> 22.38          |  1.24 -> 1.26              | 1,638,052                            |
|ViT          | MNIST       | 97.87 -> 98.87 -> 98.81 |  9.46 -> 13.04 -> 8.98     | 1,614,866 -> 87,599,250 -> 45,085,842|
|ViT          | CIFAR10     | 51.0  -> 58.70 -> 58.78 |  9.88 -> 13.98 -> 9.40     | 1,812,106 -> 88,776,970 -> 46,263,562|
|ViT          | CIFAR100    | 19.72 -> 32.87 -> 32.32 |  9.90 -> 14.24 -> 9.77     | 1,823,716 -> 88,846,180 -> 46,332,772|
|DeepViTNet   | MNIST       | 97.6  -> 98.54 -> 98.73 | 13.02 -> 16.20 -> 11.97    | 1,615,010 -> 87,601,266 -> 45,086,850|
|DeepViTNet   | CIFAR10     | 48.03 -> 55.71 -> 56.07 | 13.30 -> 19.65 -> 12.34    | 1,812,250 -> 88,778,986 -> 46,264,570|
|DeepViTNet   | CIFAR100    | 17.89 -> 29.32 -> 29.71 | 13.31 -> 19.88 -> 12.57    | 1,823,860 -> 88,848,196 -> 46,333,780|
|Hybrid       | MNIST       | 99.2  -> 99.37 -> 99.35 | 10.40 -> 12.57 -> 9.63     | 1,591,652 -> 87,275,146 -> 44,761,738|
|Hybrid       | CIFAR10     | 69.52 -> 73.02 -> 73.17 | 10.40 -> 14.86 -> 9.60     | 1,727,012 -> 88,076,106 -> 45,562,698|
|Hybrid       | CIFAR100    | 34.97 -> 45.87 -> 46.11 | 10.37 -> 15.21 -> 9.77     | 1,727,012 -> 88,145,316 -> 45,631,908|


## Experiment Four

[Planning-oriented Autonomous Driving](https://arxiv.org/pdf/2212.10156) was one of the top papers at CVPR 2023 that describes an approach for autonomous vehicles that outperforms previous state of the art methods. It uses both self and cross attention in a modular design that is trainable from end to end. We'll be taking a closer look..


![architecture](UniAD/sources/pipeline.png)


#deeplearning #attentionmechanisms #computervision