import matplotlib.pyplot as plt
import math

import matplotlib.animation as animation
import time
import numpy as np 
import argparse

Writer = animation.writers['ffmpeg']
writer = Writer(fps = 10, bitrate = 1800)

fig = plt.figure()
grid = plt.GridSpec(6, 6, hspace=0.2, wspace=0.2)

# ax2 = fig.add_subplot(3,3,1)        # Heading and calculation 
# ax3 = fig.add_subplot(3,3,3)        # Error curve
# ax1 = fig.add_subplot(3,1,2)        # Neural Net

ax1 =  fig.add_subplot(grid[1:, 0:])
ax3 = fig.add_subplot(grid[0, -1])

# ax2.axis("off")
def draw_neural_net(ax, left, right, bottom, top, layer_sizes, grad, weights, values):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    ax1.clear()
    ax1.axis('off')

    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in xrange(layer_size):
            # circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
            #                     color='w', ec='k', zorder=4)

            # # ax.add_artist(circle)
            # label = ax.annotate("cpicpi", xy=(n*h_spacing + left, layer_top - m*v_spacing), fontsize=30,
            #                     verticalalignment='center', horizontalalignment='center')
            # # text = ax.annotate("v", xy=(n*h_spacing + left , layer_top - m*v_spacing), 
            # #                     fontsize = 20, color = 'b')
            # ax.add_patch(circle)

            bbox_props = dict(boxstyle="circle,pad=0.3", fc="cyan", ec="b", lw=2)
            t = ax1.text(n*h_spacing + left, layer_top - m*v_spacing, 
                        "{}".format(round(values[n][m], 1)), ha="center", va="center", rotation=0,
                        size=12,
                        bbox=bbox_props)

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in xrange(layer_size_a):
            for o in xrange(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                slope = ((layer_top_b - o*v_spacing) - (layer_top_a - m*v_spacing))/(h_spacing)
                c_val = layer_top_a - m*v_spacing - slope*(n*h_spacing + left)
                
                theta = math.atan(slope)
                x_val = (v_spacing/2)*math.cos(theta)

                text_1 = ax1.text((n*h_spacing + left + x_val), 
                                  slope*(n*h_spacing + left + x_val) + c_val, 
                                  "{}".format(round(weights[n][m][o], 1)), fontsize = 10, color = 'g')

                text_2 = ax1.text(((n + 1)*h_spacing + left - 3/2*x_val), 
                                  slope*((n + 1)*h_spacing + left - 3/2*x_val) + c_val, 
                                  "{}".format(round(grad[n][m][o], 1)), fontsize = 10, color = 'r')
                ax1.add_artist(line)

def execute():
    grad = []
    weights = []
    values = []
    for j in range(len(layer)):
        values.append(np.random.normal(0.1, 100, layer[j]))
    for j in range(len(layer)-1):
        weights.append(np.random.normal(0.1, 100, (layer[j], layer[j+1])))
        grad.append(np.random.normal(0.1, 100, (layer[j], layer[j+1])))
    return(grad, values, weights)

def plot_error(i):
    ax3.clear()
    val = np.random.normal(0.1, 100, i+1)
    ax3.plot(range(len(val)), val)

def Animate(i):
    global grad, values, weights
    grad, values, weights = execute()
    draw_neural_net(fig.gca(), .1, .9, .1, .9, [3, 2, 1], grad, weights, values)
    plot_error(i)
    
layer = [3, 2, 1]

grad = []
weights = []
values = []

# for i in range(10):
#     grad = []
#     weights = []
#     values = []
#     for j in range(len(layer)):
#         values.append(np.random.normal(0.1, 100, layer[j]))
#     for j in range(len(layer)-1):
#         weights.append(np.random.normal(0.1, 100, (layer[j], layer[j+1])))
#         grad.append(np.random.normal(0.1, 100, (layer[j], layer[j+1])))

ani = animation.FuncAnimation(fig, Animate, frames=100, interval=200)
ani.save('Accuracy_airplane.mp4', writer = writer)
plt.show()


# fig = plt.figure(figsize=(12, 12))
# ax = fig.gca()
# ax.axis('off')
# grad = [[[1,2], [3,4], [2, 3]], [[2], [3]]]
# weights = [[[1,2], [3,4], [2, 3]], [[2], [3]]]
# values = [[1,2,3], [4, 5], [6]]
# draw_neural_net(fig.gca(), .1, .9, .1, .9, [3, 2, 1], grad, weights, values)
# plt.show()