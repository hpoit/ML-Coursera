# Visualize the data
using CSV, Plots; pyplot();
data = CSV.read("/Users/kevinliu/Documents/machine-learning-ex2/ex2/ex2data1.txt", datarow=1)
X = data[:, [1,2]]; y = data[:, 3];

pos = find(y); neg = find(iszero, y); # or neg = find(t -> t == 0, y);

scatter(xaxis=("exam 1 score", (30,100), 30:10:100))
scatter!(yaxis=("exam 2 score", (30,100), 30:10:100))
scatter!(X[pos, 1], X[pos, 2], markershape=:+, label="admitted")
scatter!(X[neg, 1], X[neg, 2], markershape=:circle, label="not admitted")

# Logit model, v0.1

# Sigmoid function
function sigmoid(z)
    1.0 ./ (1.0 .+ exp.(-z))
end

sigmoid(0) # => 0.5 as expected
z = rand(3,1); sigmoid(z) # vector
z = rand(3,3); sigmoid(z) # matrix

# Hypothesis: linearly combines X[i] and θ[i], to calculate all instances of cost()
function h(θ, X)
    z = 0
    for i in 1:length(θ)
        z += θ[i] .* X[i]
    end
    sigmoid(z)
end

# each zero θ is multiplied to each row of X
h([0,0], X)

# Cost function: for all elements of `data`, `y` defines cost variance
# y = 1 penalizes low probabilities, y = 0 penalizes high probabilities
# Penalties increase gradient differences of θ_current - average(cost(data))
function cost(θ, X, y)
    m = length(y) # number of training examples
    errorsum = 0
    for i in 1:m
        if y[i] == 1
            error = y[i] * log.(h(θ, X[i, :]))
        else y[i] == 0
            error = (1 - y[i]) * log.(1 - h(θ, X[i, :]))
        end
        errorsum += error
    end
    const constant = - 1 / m
    global J = constant * errorsum
    println("Cost is $J")
end

cost([0,0], X, y)
# => Cost is [0.693147] as expected

# θ gradient: is the partial derivative of each current θ, minus
# learning speed alpha, times the average of all costs for current θ
# Each θ has a cost
function cost_deriv(X, y, θ, J, α)
    m = length(y)
    errorsum = 0
    for i in 1:m
        error = (h(θ, Matrix(X[i, :])) - y[i]) * Matrix(X[i, :])
        errorsum += error
    end
    const constant = float(α) / float(m)
    J = constant * errorsum
end

#                  θ        α
cost_deriv(X, y, [0,0], J, 0.1)
# => 1×2 Array{Float64,2}:
#     -1.20092  -1.12628 as expected

# Gradient descent: vector in θ space from current θ to a more accurate θ
function gd(X, y, θ, α)
    m = length(y)
    θ_new = []
    const constant = α / m
    for j in 1:length(θ)
        θ_new_value = θ[j] - cost_deriv(X, y, θ, j, α)
        append!(θ_new, θ_new_value)
    end
    θ_new
end

gd(X, y, [0,0], 0.1)
# => 4-element Array{Any,1}:
 #    1.20092
 #    1.12628
 #    1.20092
 #    1.12628

# Logit model: high level function, which for iter finds gradients that map θ estimations
# to θ optimum estimations, to best represent a linear model
function logit(X, y, θ, α, iter)
    m = length(y)
    for X in 1:iter
        θ_new = gd(Matrix(X), y, θ, α)
        θ = θ_new
        if mod(X, 100) == 0
            # cost returns final hypothesis of model
            cost(X, y, θ)
            println("θ is $θ")
            println("J is $(cost(X, y, θ))")
        end
    end
end

# Test logit()
#              θ     α   iter
logit(X, y, [0, 0], 0.1, 1000)

# Test cost function
# setup data matrix
m, n = size(X);

# add intercept term to x and X_test
X = [ones(m, 1) Matrix(X)];

# initialize fitting parameters
initial_theta = zeros(n + 1, 1);

# compute and display initial cost and gradient
J, gradient = cost(initial_theta, X, y);
@printf("Cost at initial theta (zeros): %f\n", J);
@printf("Expected cost (approx): 0.693\n");
@printf("Gradient at initial theta (zeros): %f\n", gradient);
@printf("Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n");

# compute and display cost and gradient with non-zero theta
test_theta = [-24; 0.2; 0.2];
J, gradient = cost(test_theta, X, y);
@printf("Cost at test theta: %f\n", J);
@printf("Expected cost (approx): 0.218\n");
@printf("Gradient at test theta: %f\n", gradient);
@printf("Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n");

# Optimizing theta parameters
using Optim
function fmin(X, y)
    J(θ) = cost(θ, X, y)
    θ₀ = zeros(Float64, 3)
    optimize(J, θ₀)
end

fmin(X, y)

optimize(cost, zeros(Float64, 3), LBFGS())
