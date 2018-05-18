using CSV, Plots; pyplot();
data = CSV.read("/Users/kevinliu/Documents/machine-learning-ex2/ex2/ex2data1.txt", datarow=1)

# Visualize the data
X = data[:, [1,2]]; y = data[:, 3];
pos = find(y); neg = find(iszero, y); # or neg = find(t -> t == 0, y);

scatter(xaxis=("exam 1 score", (30,100), 30:10:100))
scatter!(yaxis=("exam 2 score", (30,100), 30:10:100))
scatter!(X[pos, 1], X[pos, 2], markershape=:+, label="admitted")
scatter!(X[neg, 1], X[neg, 2], markershape=:circle, label="not admitted")

# For Logit model

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

# each zero θ is multiplied to each row of X
h([0,0], X)
h([0,0,0], X)
h([-24, 0.2, 0.2], X)

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
cost([0,0,0], X, y)
# => Cost is [0.693147] as expected
cost([-24, 0.2, 0.2], X, y)
# => Cost is [0.21833] as expected

# θ gradient: is the partial derivative of each current θ, minus
# learning speed alpha, times the average of all costs for current θ
# Each θ has a cost
function cost_deriv(X, y, θ, j, α)
    m = length(y)
    errorsum = 0
    for i = 1:m, j = 1:size(X, 2)
        error = (h(θ, X[i, :]) - y[i]) .* X[i, j]
        errorsum += error
    end
    const constant = float(α) / float(m)
    J = constant * errorsum
end

cost_deriv(X, y, [0,0], 2, 0.1)
# => 1×2 Array{Float64,2}:
#     -1.20092  -1.12628 as expected in version 1
cost_deriv(X, y, [0,0,0], 2, 0.1)
cost_deriv(X, y, [-24, 0.2, 0.2], 2, 0.1)

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
gd(X, y, [0,0,0], 0.1)
gd(X, y, [-24, 0.2, 0.2], 0.1)

# Logit model: high level function, which for iter finds gradients that map θ estimations
# to θ optimum estimations, to best represent a linear model
function logit(X, y, θ, α, iter)
    for i in 1:iter
        θ_new = gd(X, y, θ, α)
        θ = θ_new
        if mod.(i, 100) == 0
            # cost returns final hypothesis of model
            cost(θ, X, y)
        end
    end
    println("θ is $θ")
    println("J is $(cost(θ, X, y))")
end

logit(X, y, [0, 0], 0.1, 1000)
logit(X, y, [0, 0, 0], 0.1, 1000)
logit(X, y, [-24, 0.2, 0.2], 0.1, 1000)

# compute and display initial cost and gradient
J, gradient = cost([0,0,0], X, y);
@printf("Cost at initial theta (zeros): %f\n", J);
@printf("Expected cost (approx): 0.693\n");
@printf("Gradient at initial theta (zeros): %f\n", gradient);
@printf("Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n");

# compute and display cost and gradient with non-zero theta
J, gradient = cost([-24, 0.2, 0.2], X, y);
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
