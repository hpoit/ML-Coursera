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
Jesse's suggestion takes visualization (and NN learning) to a more clear and
systematic level of understanding.
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
The combination of interpretability techniques provides a powerful interface.
The power of NNs lies in the hidden layers, where at each one the network discovers
a new representation of the input. The activation cube of an NN can be differently
sliced to target different activations, which are represented as abstract vectors,
and transformed to give it more meaning as a semantic dictionary.

Semantic dicts are formed by combining what a neuron looks for (vis) when it
fires (activation) and sorting them by how much they fire (activ magnitude).
This semantic dictionary finds meaningful indices and expresses an NN's learned
abstractions through canonical examples, which at times are not defined by human
language. Reducing neurons to human ideas is a lossy operation (not deep enough).

Semantic dicts in neuron terms are used here as the foundation for composable
interpretability techniques.

What the network sees in an image is the dot product maximization of the respective
activation vector. This technique applied to all vectors reveals what the network
understands of the image as a whole.

The network's understanding evolves across layers.
Scaling enables activation magnitude in each layer.

Attribution explains concept building from neuron relationships. The most common
attribution interface is the saliency map, or heatmap of the pixels that most
influence the output. This approach has two weaknesses (not discussed here).
Attribution tends to be more meaningful in high-level layers.

Channel attribution separates concepts by identifying each detector contribution
on output. Most contributing channels to output are represented by a semantic dict.

Combining space locations with channels reveals important aspects of a model, but
has two shortcomings: too much information for a human to understand from long-tail
channels and slight impacts on output, and extremely lossy aggregations.

To make interfaces human scale, breaking up activations in a meaningful way is key.
This can be done with matrix factorization, which can reveal the purpose of neuron
groups. Factorization is a function of user interface goals. NNMF reduces many
neurons to small groups that summarize an interface goal, in this case knowing
what the network detected, and the effect of each group on the output classes.

Interface space exploration depends on user goals and constraints.
"""
