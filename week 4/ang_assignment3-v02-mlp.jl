# Multilayer perceptron (thanks Mike Innes)

"""
A multilayer perceptron (MLP) is a class of feedforward artificial neural network.
An MLP consists of at least three layers of nodes. Except for the input nodes,
each node is a neuron that uses a nonlinear activation function. MLP utilizes a
supervised learning technique called backpropagation for training. Its multiple
layers and non-linear activation distinguish MLP from a linear perceptron. It
can distinguish data that is not linearly separable.
Multilayer perceptrons are sometimes colloquially referred to as "vanilla" neural
networks, especially when they have a single hidden layer.
"""

using Flux, Flux.Data.MNIST
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: repeated
# using CuArrays

# Classify MNIST digits with a simple multilayer perceptron
imgs = MNIST.images()

# Stack images into one large batch
X = hcat(float.(reshape.(imgs, :))...) |> gpu
labels = MNIST.labels()

# One-hot encode the labels
Y = onehotbatch(labels, 0:9) |> gpu
m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax) |> gpu

# see May 10 2018, week 3 notes: solving overfitting problem (underfit in this case)
# loss with L2 Ridge vecnorm regularizer (much lower accuracy)
# loss(x, y) = crossentropy(m(x), y) + sum(vecnorm, params(m))
loss(x, y) = crossentropy(m(x), y)
accuracy(x, y) = mean(argmax(m(x)) .== argmax(y))
dataset = repeated((X, Y), 200)
evalcb = () -> @show(loss(X, Y))
opt = ADAM(params(m))

Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))
"""
loss(X, Y) = 2.3444061460516727 (tracked)
loss(X, Y) = 1.592375133275002 (tracked)
loss(X, Y) = 1.024474810344954 (tracked)
loss(X, Y) = 0.7044752352587663 (tracked)
loss(X, Y) = 0.5454043818589057 (tracked)
loss(X, Y) = 0.4621782467463129 (tracked)
loss(X, Y) = 0.41190073020956053 (tracked)
loss(X, Y) = 0.37579077592255616 (tracked)
loss(X, Y) = 0.3512750966039014 (tracked)
loss(X, Y) = 0.3309072511195021 (tracked)
loss(X, Y) = 0.31448264962058164 (tracked)
loss(X, Y) = 0.3006316844443101 (tracked)
loss(X, Y) = 0.28853755259049735 (tracked)
loss(X, Y) = 0.27771179038428856 (tracked)
"""

accuracy(X, Y) # => 0.9252833333333333

# Test set accuracy
tX = hcat(float.(reshape.(MNIST.images(:test), :))...) |> gpu
tY = onehotbatch(MNIST.labels(:test), 0:9) |> gpu
accuracy(tX, tY) # => 0.9248
