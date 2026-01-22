from fktools import *
# import pyray as pr
# from mpl_toolkits.mplot3d import Axes3D

encoded_points = np.load("autoencoder_16-3-16_encodedpoints.npz")['arr_0']

encoded_points_proxy = encoded_points[[i for i in range(len(encoded_points)) if i % 1 == 0]]

print(encoded_points.shape)
print(encoded_points_proxy.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the point cloud
ax.scatter(encoded_points_proxy[:, 0], encoded_points_proxy[:, 1], encoded_points_proxy[:, 2], s=0.1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()