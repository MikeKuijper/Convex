# Convex AI

A barebones neural network library, featuring GPU accelerated matrix operations, (image) dataset file operations, and autoencoder scripting. Built around the gradient descent algorithm, it aims to minimise the cost, or, in other words, tries to function fit the neural network to a set of inputs. As such, the neural network is completely deterministic, since the outputs stay constant if the inputs stay so as well. Nothing more than an extended version of ```y = ax + b```.

It has been my pet project for the past few years, and still needs a lot of work and optimisations. It evolved from marvin.js, a similar JavaScript library, after I realised JS wasn't the way to go.

The following example demonstrates the use of the ImageClassDataset object to train a basic neural discriminator on the MNIST dataset, and save the resulting network in a file called ```mnist.bin```.

```cpp
    ConvexGPU::ImageClassDataset mnist("./t10k-images.idx3-ubyte", "./t10k-labels.idx1-ubyte", true);
    mnist.flatten();
    Convex::NeuralNetwork n({28*28, 10}, ConvexGPU::CPU);
    n.trainSequence(&mnist, 10000, "./mnist.bin");
```

The above code was used to reach a humble 78.04% accuracy on the MNIST benchmark of handwritten digits. Still not even close to human performance, but I believe it to be a significant proof of concept. NB: this code does not utilise the GPU version of the NeuralNetwork class, since it does not support all the features yet.

## Known issues
* The ConvexGPU NeuralNetwork does not yet support all the methods of the standard NeuralNetwork class. Serialising a file to disk, for example, doesn't work yet.
* Errorhandling leaves a lot to be desired
* The ConvexGPU NeuralNetwork needs a lot of optimisation, to reduce processing time.
