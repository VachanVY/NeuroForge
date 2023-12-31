# Neural Networks
## Aim of this repo
* This repo aims to make you an MLPWhiz (especially a BackpropWhiz) by creating a Neural Network from scratch **just using `torch.tensor`** (**NO using `torch`'s autograd**) and training them on the `MNIST` dataset (A dataset containing handwritten digits from 0 to 9)

## Roadmap
* The best way to go about this tutorial is to take a pencil and a piece of paper and start deriving particularly the backprop equations once you get the concept
* First we'll start off with logistic regression, which is the simpler form of MLPs just containing 1 layer and can recognize 2 classes, then scale into MLPs by adding many layers and making it recognize as many classes as you want

## Logistic Regression
* Now suppose we want to build a model that classifies a handwritten digit 9 vs any digit that is not 9
* The input to the model are the pixels of the image (which are the features) which are to be linearly transformed so that they can classify the digits, this is done with the help of learnable parameters learned from the data that we will provide
* And here we have to classify 9 vs not 9 so we only need one unit in the last layer (in MLPs we have many classes so we have `n_classes` number of units in the last layer where `n_classes` is the number of classes which will represent the probabilities for the `n_classes` given input)
* Sigmoid function: ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/a2ccf74b6142eee1c55895ba62531ba11871cf90)

    This function squishes the pre-activations (`Z`) to have a range of (0, 1)
* Then we define a threshold (which is usually 0.5), if the probabilities are above it then the digit is 9 else it's not
* Take a look at the below example\
    ![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Exam_pass_logistic_curve.svg/600px-Exam_pass_logistic_curve.svg.png)
* Forwardprop
    ```python
    X = inputs.reshape((B, H*W*1)) # (B, F=H*W) <= (B, H, W, 1) = (Batch, Height, Width, Num_Channels)
    """
    X = [
         [x00, x01, ..., x0W,
          x10, x11, ..., x1W,
          ...
          xH0, xH1, ..., xHW],
          ... (more batches of examples)
        ]
    """
    W = [[w00], # (F, 1)
         [w10],
         ...
         [wF0]]
    B = [[b0]] # (1, 1) # broadcasted and added to element in Z
    Z = X @ W + B # (B, 1) <= (B, F) @ (F, 1) + (1, 1)
    """
    Z = [[z1 = x00*w00 + x01*w10 + ... + xHW*wF0 + b0],
        ... (more batches of examples)
        ]
    """
    A = sigmoid(Z) # (B, 1)
    ```
* `Z` contains Unnormalized probabilities, the sigmoid function normalizes (range: 0-1) `Z` to get probabilities of whether the digit is 9 (the higher the probability, the more confident the model is that the digit is 9)

* Cost function: We have to penalize the model for predicting wrong values and reward it for predicting the right values

    *We want to minimize the loss to improve our model*
    
    Therefore we use the loss function: 
    ![Alt text](images/image-1.png) which just means that if

    `y_i = 1 (digit is 9)`; Loss is `-log(a_i)` which is negative log-probability of the digit is `9`; So we want to minimize `-log(a_i)` which means we want to maximize `a_i (the probability of digit being 9)` when the digit is actually 9 which is what we want

    `y_i = 0 (digit is not 9)`; Loss is `-log(1 - a_i)` which is negative log-probability of the digit not being `9`; So we want to minimize `-log(1 - a_i)` which means we want to maximize `1 - a_i (the probability of digit not being 9)` when the digit is not 9 which is again what we want 

* Now using the below equations, we'll calculate how we should change the parameters so that they incorporate the learnings from the cost function and make the model better\
    ![Alt text](images/image.png)\
    where Y is the true classes (9 or not 9)\
    We'll go through the derivations for the gradients in detail in the MLPs section below
* We'll change the parameters according to the equations below\
    `W = W - lr * dJ_dW`\
    `B = B - lr * dJ_dB`\
    ![](https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/cdp/cf/ul/g/c2/0f/ICLH_Diagram_Batch_03_21-AI-ML-GradientDescent.component.simple-narrative-xl.ts=1698244496170.png/content/adobe-cms/us/en/topics/gradient-descent/jcr:content/root/table_of_contents/body/content_section_styled/content-section-body/simple_narrative_1771421240/image)
* The above processes are usually not done taking the whole training set, this results in accurate gradients but this process is very slow as in deep learning, datasets are very large\
 Instead we take `batch_size` number of train examples from the dataset and do the above process, this results in the gradients being less accurate but this is much faster and has proved to be very much effective in practice
* We repeat the above processes for a number of `epochs`, till the model converges, see the training loop sub-section in the MLPs section for more details

---
## MLPs 
### Forward Propagation
* We stack many layers with a relu activation in-between layers and at the end add a softmax layer which calculates the probabilities given unnormalized activations
* Here unlike the sigmoid function we have `n_classes` number of units in the last layer where `n_classes` is the number of classes where each unit will represent the probabilities for each class given input
* ![forwardprop](images/forwardprop.jpg)
* Cross Entropy Loss calculation:
![loss calculation](images/loss_calculation.jpg)
We want to increase the probabilities of the true classes, therefore we minimize negative log-probs which does the same

### Back-propagation
* ![dLoss_dProba](images/dproba.jpg)
* ![softmax derivatives wrt input](images/derivative_softmax.png)
* ![dLoss_dLogits](images/dLogits.jpg)
* ![dLoss_dW3](images/dW3.jpg)
* ![dLoss_d(B3_&_H2)](images/dB3H2.jpg)
* ![dRest](images/dRest.jpg)

### Gradient Descent
* The negative gradient tells us the direction that corresponds to the steepest descent within an infinitesimally small region surrounding the current parameters
* So it's important to scale them down so that the training is stable, we do this with the help of the learning rate (lr), always keeping it less than 1 (  for very deep models we keep the lr of the order `1e-3` to `1e-5` so that the training is stable)
* We want to minimize the Loss (with the weights and biases as the parameters), we want to go down to the lowest point, the negative gradients give us the direction to the lowest point, and subtracting the parameters from their scaled-down gradients takes us downhill the Loss landscape\
![ssss](https://poissonisfish.files.wordpress.com/2020/11/non-convex-optimization-we-utilize-stochastic-gradient-descent-to-find-a-local-optimum.jpg)
* 
    ```python
    params = [w1, b1, w2, b2, w3, b3]
    grads = [dL_dw1, dL_db1, dL_dw2, dL_db2, dL_dw3, dL_db3]

    for i in range(len(params)):
        params[i] = params[i] - lr*grads[i]
    ```
---
### Training Loop
* 
    ```python
    for epoch in range(epochs):
        for step in range(steps):
            X_batch, y_batch = get_batch(X, y)
            # forward prop
            # backward prop
            # gradient descent
            ...
    ```
    In one step the model is trained on `batch_size` number of train examples\
    In one `epoch` which contains `steps` number of steps, the model is trained on all the train examples\
    This done for `epochs` number of epochs, till the model converges

---
## Results
* We obtain an accuracy of **97.39%** on the training set\
and a validation accuracy of **96.6%** on the MNIST dataset
* See the notebook to see predictions

---
# Contribution guidelines
* This repository aims to demystify neural networks, and any efforts aimed at simplification and enhancing accessibility will be incorporated
* Contributions towards changing the handwritten equations into LaTeX format are welcomed and encouraged.

---
# References
* [**Blog**: Sigmoid, Softmax and their derivatives](https://themaverickmeerkat.com/2019-10-23-Softmax/)
* [**Video**: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI)

# For Further Studies
* [**Video**: Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
* [**Book**: Deep Learning by Ian Goodfellow](https://www.deeplearningbook.org/)
