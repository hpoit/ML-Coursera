# Logit mode, v0.3

# Part 2: compute cost and gradient
using CSV
data = CSV.read("/Users/kevinliu/Documents/machine-learning-ex2/ex2/ex2data1.txt", datarow=1);

X = hcat(ones(100,1), Matrix(data[:, [1,2]])); y = Vector(data[:, 3]);

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

cost([0, 0, 0], X, y)
# Cost is 0.693147180559945
# Gradient is [-0.1, -12.0092, -11.2628]
# as expected

cost([-24, 0.2, 0.2], X, y)
# Cost is 0.21833019382659777
# Gradient is [0.042903, 2.56623, 2.6468]
# as expected

# Part 3: optimize J and gradient
using Optim

# simple minimization cases
f(x) = x[1]^2
optimize(f, [0.0,0.0])

f(x) = 100 * x[1]^2
optimize(f, [0.0,0.0])

f(x) = (100 - x[1])^2
optimize(f, zeros(2))

f(x) = x[1]^2 + x[2]^2
optimize(f, zeros(2))

f(x) = x[2] * x[1]^2
optimize(f, zeros(2))
# one-argument closure
optimize(x -> f(x), zeros(2))

# minimize J
J(θ) = (-y' * log.(sigmoid(X * θ)) - (1 - y') * log.(1 - sigmoid(X * θ))) / length(y)
optimize(J, zeros(3), BFGS())

thetas = Optim.minimizer(optimize(J, zeros(3), ConjugateGradient()))
# => 3-element Array{Float64,1}:
#     -25.1614
#       0.206232
#       0.201472
# thetas as expected

Optim.minimum(optimize(J, zeros(3), BFGS()))
# => 0.2034977015894402
# θ for J as expected

# Comparing optimizers
"""
optimize(J, zeros(3), ConjugateGradient())
Results of Optimization Algorithm
 * Algorithm: Conjugate Gradient
 * Starting Point: [0.0,0.0,0.0]
 * Minimizer: [-25.161375399086644,0.20623204812650806, ...]
 * Minimum: 2.034977e-01
 * Iterations: 867
 * Convergence: true
   * |x - x'| ≤ 1.0e-32: false
     |x - x'| = 1.46e-10
   * |f(x) - f(x')| ≤ 1.0e-32 |f(x)|: true
     |f(x) - f(x')| = 0.00e+00 |f(x)|
   * |g(x)| ≤ 1.0e-08: false
     |g(x)| = 1.21e-08
   * Stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 1747
 * Gradient Calls: 966

optimize(J, zeros(3), LBFGS())
Results of Optimization Algorithm
 * Algorithm: L-BFGS
 * Starting Point: [0.0,0.0,0.0]
 * Minimizer: [-25.161334539967783,0.20623172118688082, ...]
 * Minimum: 2.034977e-01
 * Iterations: 16
 * Convergence: true
   * |x - x'| ≤ 1.0e-32: false
     |x - x'| = 2.36e-06
   * |f(x) - f(x')| ≤ 1.0e-32 |f(x)|: false
     |f(x) - f(x')| = 2.81e-14 |f(x)|
   * |g(x)| ≤ 1.0e-08: true
     |g(x)| = 9.28e-10
   * Stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 58
 * Gradient Calls: 58

optimize(J, zeros(3), BFGS())
Results of Optimization Algorithm
 * Algorithm: BFGS
 * Starting Point: [0.0,0.0,0.0]
 * Minimizer: [-25.161334548093443,0.20623172126990053, ...]
 * Minimum: 2.034977e-01
 * Iterations: 14
 * Convergence: true
   * |x - x'| ≤ 1.0e-32: false
     |x - x'| = 2.19e-06
   * |f(x) - f(x')| ≤ 1.0e-32 |f(x)|: false
     |f(x) - f(x')| = 7.82e-14 |f(x)|
   * |g(x)| ≤ 1.0e-08: true
     |g(x)| = 1.47e-09
   * Stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 47
 * Gradient Calls: 47
 """

# Plot boundary (still part 3)
# Plot first: decision boundary with θ
using CSV, StatPlots, Plots; pyplot();
data = CSV.read("/Users/kevinliu/Documents/machine-learning-ex2/ex2/ex2data1.txt", datarow=1);

X = hcat(ones(100,1), Matrix(data[:, [1,2]])); y = Vector(data[:, 3]); # works for plot
#X = data[:, [1,2]]; y = data[:, 3]; # works for plot

# map two input features to quadratic features used in regularization
function mapfeature(x1, x2)
    degree = 6 # why 6?
    out = ones(size(x1[:,1]))
    for i = 1:degree
        for j = 0:i
            out[:, end + 1] = [x1 .^ [i - j]] .* [x2 .^ j]
        end
    end
end

function plotdboundary(θ, X, y)
    # if second dimension of X (feature columns) is <= 3
    if size(X, 2) <= 3 # => 3, yes, for X => 100x3 Array{Union{Float64, Missings.Missing},2}
        # two endpoints
        plot_xaxis = [minimum(X[:,2])-2,  maximum(X[:,2])+2]
        # compute decision boundary line
        plot_yaxis = (-1 ./ θ[3]) .* (θ[2] .* plot_xaxis + θ[1])
        plot(plot_xaxis, plot_yaxis, label="decision boundary")
    else # if size(X, 2) > 3 (if second dimension of X is > 3)
        # grid range
        u = linspace(-1, 1.5, 50)
        v = linspace(-1, 1.5, 50)
        z = zeros(length(u), length(v));
        # Evaluate z = θ * x over the grid
        for i = 1:length(u)
            for j = 1:length(v)
                z[i,j] = mapfeature(u[i], v[j]) * θ
            end
        end
        # transpose z before calling contour
        z = z'
        # Plot z = 0, specify the range [0, 0]
        # Contour should maximize inter-class distance
        contour(u, v, z, [0, 0], lineWidth=2)
    end
end

plotdboundary([0.203,0.203,0.203], X, y)

# Scatter second: X
X = data[:, [1,2]]; y = data[:, 3]; # works for scatter

X = Matrix(data[:, [1,2]]); y = Vector(data[:, 3]); # works for scatter

pos = find(y); neg = find(iszero, y); # or neg = find(t -> t == 0, y);

scatter(xaxis=("exam 1 score", (30,100), 30:10:100))
scatter!(yaxis=("exam 2 score", (30,100), 30:10:100))
scatter!(X[pos, 1], X[pos, 2], markershape=:+, label="admitted")
scatter!(X[neg, 1], X[neg, 2], markershape=:circle, label="not admitted")
