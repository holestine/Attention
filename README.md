# Attention
This is a collection of links, descriptions and experiments performed while doing a deep dive on attention mechanisms.

## Experiment One
The first resource I found that significantly improved my understanding of the paper [Attention Is All You Need](https://arxiv.org/pdf/1706.03762 "Attention Is All You Need") was from the write up [Transformers from Scratch in PyTorch](https://fkodom.substack.com/p/transformers-from-scratch-in-pytorch "Transformers from Scratch in PyTorch Write Up"). The author takes excerpts from "Attention Is All You Need" and implements the Scaled Dot-Product Attention and Multi-Head Attention Mechanism one piece at a time using PyTorch. There is also a [code repo](https://github.com/fkodom/transformer-from-scratch "Transformers from Scratch in PyTorch Repo") with the implementation with unit tests done in pytest. I've forked the repo here and add the file [transformer-from-scratch/test.py](https://github.com/holestine/transformer-from-scratch/blob/main/test.py) to make it easy to step through with VSCode. 

## Experiment Two
Since I come from a Computer Vision background I was interested to see how well transformers work with images and came across the [repo for vit-pytorch](https://github.com/lucidrains/vit-pytorch "vit-pytorch") which has implementations for several different attention mechanisms. I then copied the [PyTorch MNIST Sample](https://github.com/pytorch/examples/blob/main/mnist/main.py) to [mnist.py](./mnist.py) and replaced the CNN model with a single multi-headed vision transformer.

```
self.model = ViT(
                        image_size = 784,
                        patch_size = 4,
                        num_classes = 10,
                        dim = 1024,
                        depth = 6,
                        heads = 16,
                        mlp_dim = 2048,
                        channels=1,
                        dropout = 0.1,
                        emb_dropout = 0.1
                        )
```
