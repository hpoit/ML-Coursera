# A. Ng's process
# Part 1: load and visualize data
# Part 2: load parameters
# Part 3: compute cost (feedforward)
# Part 4: regularization
# Part 5: sigmoid gradient
# Part 6: initialize parameters
# Part 7: backpropagation
# Part 8: train NN
# Part 9: visualize trained parameters (weights)
# Part 10: predict

# Flux's MLP https://github.com/hpoit/ML-Coursera/blob/master/week%204/ang_assignment3-v02-mlp.jl
# covers Parts 1-10 except 9, which has yet to be built for Flux
# Jesse Bettencourt (https://github.com/jessebett) suggested I read
# https://distill.pub/2017/feature-visualization/
"""
For NN interpretability research, a new network for visualizing neural networks
will have to be built: https://github.com/tensorflow/lucid/blob/master/notebooks/tutorial.ipynb
"""
"""
Distilling the Distilled, for a Julia implementation:
Jesse's suggestion takes visualization to a whole new level.
In the article, feature visualization (what a neuron looks for) is done through
optimization for accuracy and optimized diversification for better feature definition
(what causes a neuron to fire). To understand neural nets, the article observes
combinations of optimized and interpolated pair neurons working together to
represent images.

Optimization simultaneously generates noise and high frequency patterns,
which are traditionally reduced by the three intermediate regularization
families, trade-off being accurate correlations v. realistic examples.
However, regularization as is known only supresses noise, so combining it with
gradient transformation (preconditioning) further improves visualization.
"""
"""
Next: https://distill.pub/2018/building-blocks/
"""
