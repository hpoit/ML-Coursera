# Intial setup of parameters
input_layer_size  = 400  # 20x20 Input Images of Digits
num_labels = 10          # for digits 0-9; digit 0 is mapped to label 10

# Part 1: loading and visualizing
# Read 5000 y training examples of digits with corresponding vectors x

# MNIST.jl provides access by the i-th image x or label (example) y
# MNIST contains 60,000 images, each with 784 features
# trainfeatures() and trainlabel() call up to 60000
using MNIST
x = trainfeatures(5000) # => 784-element Array{Float64,1} (feature vector of image 5000)
y = trainlabel(5000) # => 2.0 (label of image 5000: digit 2)

# for 100 random data points of matrix X, consider
X = rand(10, 15)
m = size(X, 1)
rand_ind = randperm(m); rand_ind[1:5, :] # rows 1-5 of randomized indices of first column
X[rand_ind[1:5,:]]

# for 100 random data points of vector x, do
X, y = traindata() # X => 784x60000 Array{Float64,2}, y => 60000-element Array{Float64,1}
rand_indices = randperm(size(X,1)) # => 784 element vector of randomized indices
subset = X[rand_indices[1:100], :] # => 100×60000 matrix of randomized indices

# Part 2a: vectorized and regularized logit for one-vs-all classification of dataset

# Sigmoid function
function sigmoid(z)
    1.0 ./ (1.0 .+ exp.(-z))
end

sigmoid(0) # => 0.5
z = rand(3,1); sigmoid(z) # vector
z = rand(3,3); sigmoid(z) # matrix

# Hypothesis: linearly combines X[i] and θ[i], to calculate all instances of cost()
function h(θ, X)
    z = 0
    for i in 1:length(θ)
        z += θ[i] .* X[i, :]
    end
    sigmoid(z)
end

h([-24, 0.2, 0.2], X)

# Vectorized cost function
function cost(θ, X, y)
    hx = sigmoid(X * θ)
    m = length(y)
    global J = (-y' * log.(hx) - (1 - y') * log.(1 - hx)) / m
    global grad = X' * (hx - y) / m
    println("Cost is $J")
    println("Gradient is $grad")
end

# test cost function
theta_t = [-2, -1, 1, 2]
X_t = [ones(5, 1) reshape(1:15, 5, 3) / 10]
y_t = [[1, 0, 1, 0, 1] >= 0.5]
lambda_t = 3

cost(theta_t, X_t, y_t, # missing lambda_t)

# Part 2b: one-vs-all training
function one_vs_all end

lambda = 0.1
all_theta = one_vs_all(X, y, num_labels, lambda)
