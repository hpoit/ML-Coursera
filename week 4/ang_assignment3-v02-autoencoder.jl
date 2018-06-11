# Autoencoder (thanks Mike Innes)

"""
An autoencoder is a type of artificial neural network used to learn efficient
data codings in an unsupervised manner. The aim of an autoencoder is to learn
a representation (encoding) for a set of data, typically for the purpose of
dimensionality reduction.
"""

using Flux, Flux.Data.MNIST
using Flux: @epochs, onehotbatch, argmax, mse, throttle
using Base.Iterators: partition
using Juno: @progress
# using CuArrays

# Encode MNIST images as compressed vectors that can later be decoded back into images
imgs = MNIST.images()

# Partition into batches of size 1000
data = [float(hcat(vec.(imgs)...)) for imgs in partition(imgs, 1000)]
data = gpu.(data)
N = 32 # Size of the encoding output layer

# The encoder/decoder network can be made larger
# Encoder output is based on input
# Here, the input dimension is 28^2 and encoder output dimension is 32,
# which implies the encoding is a compressed representation.
# The compressed encoding can also be made lossy
encoder = Dense(28^2, N, relu) |> gpu # layer 1
decoder = Dense(N, 28^2, relu) |> gpu # layer 2
m = Chain(encoder, decoder)
loss(x) = mse(m(x), x)
evalcb = throttle(() -> @show(loss(data[1])), 5)
opt = ADAM(params(m))
@epochs 10 Flux.train!(loss, zip(data), opt, cb = evalcb)

# Sample output
using Images, QuartzImageIO
img(x::Vector) = Gray.(reshape(clamp.(x, 0, 1), 28, 28))
function sample()
    before = [imgs[i] for i in rand(1:length(imgs), 20)] # 20 random digits
    after = img.(map(x -> cpu(m)(float(vec(x))).data, before)) # Before and after images
    hcat(vcat.(before, after)...) # Stack them all together
end

cd("/Users/kevinliu/Google\ Drive/ML\ Coursera/week\ 4")
save("autoencoded.png", sample())
# png available at https://github.com/hpoit/ML-Coursera/blob/master/week%204/autoencoded.png
