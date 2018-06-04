# Autoencoder (thanks Mike Innes)

using Flux, Flux.Data.MNIST
using Flux: @epochs, onehotbatch, argmax, mse, throttle
using Base.Iterators: partition
using Juno: @progress
# using CuArrays

# Encode MNIST images as compressed vectors that can later be decoded back into
# images.

imgs = MNIST.images()

# Partition into batches of size 1000
data = [float(hcat(vec.(imgs)...)) for imgs in partition(imgs, 1000)]
data = gpu.(data)

N = 32 # Size of the encoding

# You can try to make the encoder/decoder network larger
# Also, the output of encoder is a coding of the given input.
# In this case, the input dimension is 28^2 and the output dimension of
# encoder is 32. This implies that the coding is a compressed representation.
# We can make lossy compression via this `encoder`.
encoder = Dense(28^2, N, relu) |> gpu
decoder = Dense(N, 28^2, relu) |> gpu

m = Chain(encoder, decoder)

loss(x) = mse(m(x), x)

evalcb = throttle(() -> @show(loss(data[1])), 5)
opt = ADAM(params(m))

@epochs 10 Flux.train!(loss, zip(data), opt, cb = evalcb)

# Sample output

using Images, QuartzImageIO

img(x::Vector) = Gray.(reshape(clamp.(x, 0, 1), 28, 28))

function sample()
  # 20 random digits
  before = [imgs[i] for i in rand(1:length(imgs), 20)]
  # Before and after images
  after = img.(map(x -> cpu(m)(float(vec(x))).data, before))
  # Stack them all together
  hcat(vcat.(before, after)...)
end

cd("/Users/kevinliu/Google\ Drive/ML\ Coursera/week\ 4")

save("autoencoded.png", sample())
# available at https://github.com/hpoit/ML-Coursera/blob/master/week%204/autoencoded.png
