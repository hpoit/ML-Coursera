# Linear regression

# part 1: basic function, A = eye(5), trivial

# part 2: scatter initial data
using CSV
data = CSV.read("/Users/kevinliu/Documents/machine-learning-ex1/ex1/ex1data1.txt", datarow=1);

x = convert(Vector{Float64},data[:, 1]); y = convert(Vector{Float64},data[:, 2]);

m = length(y); # number of training examples

using Plots

scatter(x, y, xlabel="city population in thousands", ylabel="profit in thousands")
# graph as "scatter population and profit.png"

# part 3: cost and gradient descent
X = hcat(ones(m, 1), x);

# define cost function J
J(θ) = inv(2m) * sum(abs2, X * θ - y)

# test cost function J
J([0.0, 0.0])
# => 32.072733877455676 as expected

J([-1, 2])
# => 54.24245508201238 as expected

# minimize J
using Optim; res = optimize(J, zeros(2), BFGS())
"""
Results of Optimization Algorithm
 * Algorithm: BFGS
 * Starting Point: [0.0,0.0]
 * Minimizer: [-3.895780878170897,1.19303364420903]
 * Minimum: 4.476971e+00
 * Iterations: 2
 * Convergence: true
   * |x - x'| ≤ 1.0e-32: false
     |x - x'| = 3.97e+00
   * |f(x) - f(x')| ≤ 1.0e-32 |f(x)|: false
     |f(x) - f(x')| = 3.20e-01 |f(x)|
   * |g(x)| ≤ 1.0e-08: true
     |g(x)| = 2.64e-09
   * Stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 7
 * Gradient Calls: 7
 """

# plot linear regression onto scatter
θ = Optim.minimizer(res)
feature_xaxis = collect(extrema(x))
regress_yaxis = (1 ./ θ[2]) .* (feature_xaxis + θ[1])
plot!(feature_xaxis, regress_yaxis, label = "linear regression")
# see "linear regression onto scatter.png"

# hypothesis (prediction function) (not on assignment)
h(θ, X) = X * θ
ŷ = h(θ, X);
