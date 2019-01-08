import matplotlib.pyplot as plt



fig = plt.figure(figsize=(12, 12))
ax = fig.gca()

line=plt.Line2D([0,1], [1,0], c='k')
ax.add_artist(line)
ax.axis('off')
plt.show()