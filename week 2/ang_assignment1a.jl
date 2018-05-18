# Linear regression

# part 1: basic function, A = eye(5), trivial

# part 2: plotting
julia> using CSV

julia> data = CSV.read("/Users/kevinliu/Downloads/machine-learning-ex1/ex1/ex1data1.txt", datarow=1)
97×2 DataFrames.DataFrame
│ Row │ Column1 │ Column2 │
├─────┼─────────┼─────────┤
│ 1   │ 6.1101  │ 17.592  │
│ 2   │ 5.5277  │ 9.1302  │
│ 3   │ 8.5186  │ 13.662  │
⋮
│ 95  │ 8.2934  │ 0.14454 │
│ 96  │ 13.394  │ 9.0551  │
│ 97  │ 5.4369  │ 0.61705 │

julia> X = data[:, 1]
97-element Array{Union{Float64, Missings.Missing},1}:
  6.1101
  5.5277
  8.5186
  7.0032
  ⋮
  5.3054
  8.2934
 13.394
  5.4369

julia> y = data[:, 2]
97-element Array{Union{Float64, Missings.Missing},1}:
 17.592
  9.1302
 13.662
 11.854
  ⋮
  0.14454
  9.0551
  0.61705

julia> m = length(y); # number of training examples

julia> using Plots

julia> scatter(X, y, xlabel="profit in thousands", ylabel="city population in thousands")

# part 3: cost and gradient descent

# add matrix X to a ones vector (notice reverse order)
julia> X2 = hcat(ones(m, 1), data[:, 1])
97×2 Array{Union{Float64, Missings.Missing},2}:
 1.0   6.1101
 1.0   5.5277
 1.0   8.5186
 ⋮

 1.0   8.2934
 1.0  13.394
 1.0   5.4369

# initialize parameters
julia> θ = zeros(2, 1)
2×1 Array{Float64,2}:
 0.0
 0.0

# define iterations and alpha
julia> iterations = 1500; alpha = 0.01;

# define cost function J
julia> function J(X, y, θ)
           # X is design matrix containing training examples
           # y is vector of class labels
           m = size(X, 1) # number of training examples
           predictions = X*θ # hypotheses predictions on all m
           sqrErrors = (predictions - y).^2 # squared errors
           J = 1/(2*m) * sum(sqrErrors)
       end

J (generic function with 1 method)

# test cost function J
julia> J(X2, y, θ)
32.072733877455676 # as expected

julia> J(X2, y, [-1; 2])
54.24245508201238 # as expected

# GD option 1
# define gradient descent
# h_θ(x) = θ₀ + θ₁*x is the hypothesis function
using CSV
function error_rate(θ₀, θ₁, data)
    totalError = 0
    for i in 1:length(data)
         x = data[i, 1]
         y = data[i, 2]
         totalError += (y - (θ₁ * x + θ₀)) ^ 2
    end
    return totalError / float(length(data))
end

function step_gradient(current_θ₀, current_θ₁, data, α)
    gradient_θ₀ = 0
    gradient_θ₁ = 0
    N = float(length(data))
    for i in 1:length(data)
        x = data[i, 1]
        y = data[i, 2]
        gradient_θ₀ += -(2/N) * (y - ((current_θ₁ * x) + current_θ₀))
        gradient_θ₁ += -(2/N) * x * (y - ((current_θ₁ * x) + current_θ₀))
    end
    new_θ₀ = current_θ₀ - (α * gradient_θ₀)
    new_θ₁ = current_θ₁ - (α * gradient_θ₁)
    new_θ₀, new_θ₁
end

function gd_runner(data, starting_θ₀, starting_θ₁, α, iterations)
    θ₀ = starting_θ₀
    θ₁ = starting_θ₁
    for i in 1:iterations
        θ₀, θ₁ = step_gradient(θ₀, θ₁, data, α)
    end
    θ₀, θ₁
end

# run gradient descent
function run()
    data = CSV.read("/Users/kevinliu/Downloads/machine-learning-ex1/ex1/ex1data1.txt", datarow=1)
    α = 0.01
    initial_θ₀ = 0
    initial_θ₁ = 0
    iterations = 1500
    println("Starting gradient descent at θ₀ = $initial_θ₀, θ₁ = $initial_θ₁, error = $(error_rate(initial_θ₀, initial_θ₁, data))")
    (θ₀, θ₁) = gd_runner(data, initial_θ₀, initial_θ₁, α, iterations)
    print("After $iterations iterations, θ₀ = $θ₀, θ₁ = $θ₁, error = $(error_rate(θ₀, θ₁, data))")
end

# possible thetas are -3.6303 and 1.1664
# INCORRECT THETAS WHEN COMPARED TO SCATTER() ON LINE 47
run()
Starting gradient descent at θ₀ = 0, θ₁ = 0, error = 196.41950802
After 1500 iterations, θ₀ = -4.090664703327588e59, θ₁ = -2.386107508525324e60, error = 2.0478943723053065e122

# GD option 2
using CSV
data = CSV.read("/Users/kevinliu/Downloads/machine-learning-ex1/ex1/ex1data1.txt", datarow=1)
function gd(data, α, epsilon, iterations)
    i = 1
    x = data[i,1]
    y = data[i,2]
    converged = false
    m = length(data[:,1]) # length of x1
    θ₀ = 0
    θ₁ = 0
    J = sum([(θ₀ + θ₁ * x - y)^2 for i in m])
    while true
        # for each sample, compute gradient
        θ₀_grad = 1.0/m * sum([(θ₀ + θ₁ * x - y) for i in m])
        θ₁_grad = 1.0/m * sum([(θ₀ + θ₁ * x - y) * x for i in m])
        # update θ_temp
        θ₀_temp = θ₀ - α * θ₀_grad
        θ₁_temp = θ₁ - α * θ₁_grad
        # update θ
        θ₀ = θ₀_temp
        θ₁ = θ₁_temp
        # mean squared error
        mse = sum([(θ₀ + θ₁ * x - y)^2 for i in m])
        # if epsilon close, clock out regardless of iter left (error tolerance)
        if abs(J - mse) <= epsilon
            println("Converged with $i iterations")
            converged = true
            break
        end
        J = mse # update error
        i += 1 # update iterations
        if i == iterations
            println("Stopped at $i iterations")
            break
        end
    end
    println("θ₀ = $θ₀, θ₁ = $θ₁")
end

# possible thetas are -3.6303 and 1.1664
gd(data, 0.01, 0.0001, 1500)
Converged with 1277 iterations
θ₀ = 0.4559997241594341, θ₁ = 2.786203914586563
