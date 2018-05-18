# Logit mode, v0.3

using CSV, Plots; pyplot();
data = CSV.read("/Users/kevinliu/Documents/machine-learning-ex2/ex2/ex2data1.txt", datarow=1)

X = hcat(ones(100,1), Matrix(data[:, [1,2]]))
y = Vector(data[:, 3])

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
    hx = h(θ, X)
    m = length(X)
    J = (-y' .* log(hx) .- (1 .- y') .* log(1 .- hx)) ./ m
    grad = X' .* (hx .- y) ./ m
    println("Cost is $J")
    println("Gradient is $grad")
end

cost([-24, 0.2, 0.2], X, y)
