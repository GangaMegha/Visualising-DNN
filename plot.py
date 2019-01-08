################## CODE for generating video ###############

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np 
import argparse


########################### Offline_Accuracy_Airplane ############################

Writer = animation.writers['ffmpeg']
writer = Writer(fps = 30, bitrate = 1800)

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def Animate(i):
    ax1.clear()

    ax1.plot(frame_xA[:i+1], accuracy_A[:i+1], color = 'b', label = "ApproxNet", lw = 1)
    ax1.plot(frame_xM[:i+1], accuracy_M[:i+1], color = 'r', label = "MCDNN", lw = 1)

    ax1.set_xlabel('# Frames') 
    ax1.set_ylim([-2,102])
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend(loc='upper center')
    ax1.grid()

ani = animation.FuncAnimation(fig, Animate, interval=200)
ani.save('Accuracy_airplane.mp4', writer = writer)
plt.show()

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', action="store", dest="lr", default=0.01, type = float)
    parser.add_argument('--momentum', action="store", dest="momentum", default=0.5, type = float)
    parser.add_argument('--num_hidden', action="store", dest="num_hidden", default=3, type = int)
    parser.add_argument('--sizes', action="store", dest="sizes", type = str)
    parser.add_argument('--activation', action="store", dest="activation", default="sigmoid", type = str)
    parser.add_argument('--loss', action="store", dest="loss", default="sq", type = str)
    parser.add_argument('--opt', action="store", dest="opt", default="adam", type = str)
    parser.add_argument('--batch_size', action="store", dest="batch_size", default=20, type = int)
    # parser.add_argument('--anneal', action="store", dest="anneal")
    parser.add_argument('--save_dir', action="store", dest="save_dir", default="", type = str)
    parser.add_argument('--expt_dir', action="store", dest="expt_dir", default="", type = str)
    parser.add_argument('--train', action="store", dest="train", default="train.csv", type = str)
    # parser.add_argument('--test', action="store", dest="test", default="test.csv", type = str)
    # parser.add_argument('--val', action="store", dest="val", default="scale_val.csv", type = str)
    parser.add_argument('--pretrain', action="store", dest="pretrain", default="false", type = str)

    arg_val = parser.parse_args()
  
  # Read the data and train the network
    get_data(arg_val)

parse()
