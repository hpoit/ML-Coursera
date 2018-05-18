# Linear regression

# part 3b: plotting linear fit
using Plots, CSV
data = CSV.read("/Users/kevinliu/Downloads/machine-learning-ex1/ex1/ex1data1.txt", datarow=1)
m = length(data[:, 2])
X = hcat(ones(m, 1), data[:, 1]);
y = data[:, 2]
#thetasA = [-4.090664703327588; -2.386107508525324]; # incorrect when compared to scatter()
thetasB = [0.4559997241594341; 2.786203914586563];
thetasNg = [-3.6303; 1.1664];
#plot(X[:,2], X * thetasA) # incorrect when compared to scatter()
plot(X[:,2], X * thetasB)
plot(X[:,2], X * thetasNg)

# part 4: visualizing J(θ₀, θ₁)
# grid over which to calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

# initialize J_vals to a matrix of zeros
J_vals = zeros(length(theta0_vals), length(theta1_vals));

# Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals[i]; theta1_vals[j]]
	  #J_vals[i,j] = computeCost(X, y, t)
      J_vals[i,j] = sum((X * t - y).^2) / (2 * m)
    end
end

# Plot 3D surface
# x and y axis
plot(theta0_vals, theta1_vals, J_vals, xaxis=("theta0"), yaxis=("theta1"),
        m=(10, 0.8, :blues, stroke(0)), cbar=true, w=5)
# z axis
plot!(zeros(50),zeros(50),1:50,w=10)

# Plot contour
# use https://github.com/GiovineItalia/Gadfly.jl/issues/1144#issuecomment-388225727
contour(theta0_vals, theta1_vals, J_vals, xaxis=("theta0"), yaxis=("theta1"))
