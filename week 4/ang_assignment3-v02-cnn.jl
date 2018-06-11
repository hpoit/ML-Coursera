# Convolutional neural network (thanks Mike Innes)

"""
In machine learning, a convolutional neural network (CNN, or ConvNet) is a class
of deep, feed-forward artificial neural networks, most commonly applied to
analyzing visual imagery.
CNNs use a variation of multilayer perceptrons designed to require minimal
preprocessing. They are also known as shift invariant or space invariant
artificial neural networks (SIANN), based on their shared-weights architecture
and translation invariance characteristics.
Convolutional networks were inspired by biological processes in that the
connectivity pattern between neurons resembles the organization of the
animal visual cortex.
"""

using Flux, Flux.Data.MNIST
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: repeated, partition
# using CuArrays

# Classify MNIST digits with a convolutional network
imgs = MNIST.images()
labels = onehotbatch(MNIST.labels(), 0:9)

# Partition into batches of size 1000
train = [(cat(4, float.(imgs[i])...), labels[:,i])
         for i in partition(1:60_000, 1000)]
train = gpu.(train)

# Prepare test set (first 1000 images)
tX = cat(4, float.(MNIST.images(:test)[1:1000])...) |> gpu
tY = onehotbatch(MNIST.labels(:test)[1:1000], 0:9) |> gpu
m = Chain(
  Conv((2,2), 1=>16, relu), # layer 1 with relu activation function
  x -> maxpool(x, (2,2)),
  Conv((2,2), 16=>8, relu), # layer 2 with relu activation function
  x -> maxpool(x, (2,2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(288, 10), softmax) |> gpu # layer 3
m(train[1][1])
loss(x, y) = crossentropy(m(x), y)
accuracy(x, y) = mean(argmax(m(x)) .== argmax(y))
evalcb = throttle(() -> @show(accuracy(tX, tY)), 10)
opt = ADAM(params(m))
Flux.train!(loss, train, opt, cb = evalcb)
"""
accuracy(tX, tY) = 0.126
accuracy(tX, tY) = 0.126
accuracy(tX, tY) = 0.178
accuracy(tX, tY) = 0.281
accuracy(tX, tY) = 0.366
accuracy(tX, tY) = 0.533
accuracy(tX, tY) = 0.666
accuracy(tX, tY) = 0.693
"""
