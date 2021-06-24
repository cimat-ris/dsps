import numpy as np
import os

# search for files in a directory
files = []
for root, dirs, f in os.walk("./training/"):
    files.append(f)
files = files[0]

# Search for all basename* .npy files
subfiles = [s for s in files if "151" in s and ".npy" in s]
subfiles.sort()
print(subfiles)
# print(len(subfiles))

# loop tracker
k = 1

# opening all the files in the specified range
for i in range(0, 1):
    with open("./training/" + subfiles[i], 'rb') as f:
        x_training = np.load(f)
        y_training = np.load(f)
        print("Loop number:", k)
        # print(x_training.shape)
        # print(y_training.shape)
        streamlines = x_training
        all_labels = y_training


    # function to convert each streamline into a fibermap
    def get_fibermap(all_trajs, n):
        fiber_map = np.zeros([2 * n, 2 * n, 3])  # define empty map for each streamline
        all_fibre_map = np.zeros([len(all_trajs), 2 * n, 2 * n, 3])  # to store all maps

        for j in range(len(all_trajs)):

            data = all_trajs[j]  # choose one streamline

            for i in range(3):  # for each dimension in streamline
                stream = data[:, i]
                stream_rev = stream[::-1]  # reverse

                block1 = np.concatenate((stream, stream_rev), axis=0)  # build blocks
                block2 = np.concatenate((stream_rev, stream), axis=0)

                cell = np.vstack((block1, block2))  # stack vertically

                fiber_slice = np.tile(cell, (n, 1))  # create fiber map

                fiber_map[:, :, i] = fiber_slice  # assign to map for each dimension

            all_fibre_map[j, :, :, :] = fiber_map  # save all maps from all streamlines

        return all_fibre_map


    # run function with streamline data
    map1 = get_fibermap(streamlines, 20)
    # print(map1.shape)

    k += 1
    print("executed")
    print(" ")

    np.save("./map1.npy", map1, allow_pickle = True)
    np.save("./label1.npy", all_labels, allow_pickle = True)
