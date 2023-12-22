import numpy as np
import matplotlib.pyplot as plt



def grid_mask(height,width,grid_num=2):
    mask = np.zeros((height, width))
    height_boardary = np.arange(0,height, step= int(height/grid_num)+1)
    width_boardary = np.arange(0, width, step=int(width / grid_num)+1)


    count = 0
    for i in range(len(height_boardary)-1):
        for j in range(len(width_boardary)-1):
            mask[height_boardary[i]:height_boardary[i+1], width_boardary[j]:width_boardary[j+1]] = count
            count += 1
        mask[height_boardary[i]:height_boardary[i + 1], width_boardary[-1]:] = count
        count += 1

    for j in range(len(width_boardary) - 1):
        mask[height_boardary[-1]:, width_boardary[j]:width_boardary[j + 1]] = count
        count += 1

    mask[height_boardary[-1]:, width_boardary[-1]:] = count
    count += 1

    return mask.astype(int), count


def grid_mask_plus(height,width,grid_num_height=2,grid_num_width=2):
    mask = np.zeros((height, width))
    height_boardary = np.arange(0,height, step= int(height/grid_num_height)+1)
    width_boardary = np.arange(0, width, step=int(width / grid_num_width)+1)


    count = 0
    for i in range(len(height_boardary)-1):
        for j in range(len(width_boardary)-1):
            mask[height_boardary[i]:height_boardary[i+1], width_boardary[j]:width_boardary[j+1]] = count
            count += 1
        mask[height_boardary[i]:height_boardary[i + 1], width_boardary[-1]:] = count
        count += 1

    for j in range(len(width_boardary) - 1):
        mask[height_boardary[-1]:, width_boardary[j]:width_boardary[j + 1]] = count
        count += 1

    mask[height_boardary[-1]:, width_boardary[-1]:] = count
    count += 1

    return mask.astype(int), count



def grid_mask_3D_plus(times, height,width,grid_num_times=2,grid_num_height=2,grid_num_width=2):
    mask = np.zeros((times, height, width))
    times_boardary = np.arange(0,times, step= int(times/grid_num_times)+1)
    height_boardary = np.arange(0,height, step= int(height/grid_num_height)+1)
    width_boardary = np.arange(0, width, step=int(width / grid_num_width)+1)


    count = 0
    for t in range(len(times_boardary) - 1):
        for i in range(len(height_boardary)-1):
            for j in range(len(width_boardary)-1):
                mask[times_boardary[t]:times_boardary[t+1],
                        height_boardary[i]:height_boardary[i+1], width_boardary[j]:width_boardary[j+1]] = count
                count += 1
            mask[times_boardary[t]:times_boardary[t+1], height_boardary[i]:height_boardary[i + 1], width_boardary[-1]:] = count
            count += 1

        for j in range(len(width_boardary) - 1):
            mask[times_boardary[t]:times_boardary[t+1], height_boardary[-1]:, width_boardary[j]:width_boardary[j + 1]] = count
            count += 1

        mask[times_boardary[t]:times_boardary[t+1], height_boardary[-1]:, width_boardary[-1]:] = count
        count += 1

    for i in range(len(height_boardary) - 1):
        for j in range(len(width_boardary) - 1):
            mask[times_boardary[-1]:, height_boardary[i]:height_boardary[i + 1],
                                        width_boardary[j]:width_boardary[j + 1]] = count
            count += 1

        mask[times_boardary[-1]:, height_boardary[i]:height_boardary[i + 1], width_boardary[-1]:] = count
        count += 1

    for j in range(len(width_boardary) - 1):
        mask[times_boardary[-1]:, height_boardary[-1]:, width_boardary[j]:width_boardary[j + 1]] = count
        count += 1



    mask[times_boardary[-1]:, height_boardary[-1]:, width_boardary[-1]:] = count
    count += 1

    return mask.astype(int), count






if __name__ == '__main__':
    labels_out,count = grid_mask_plus(240,320, grid_num_height=4,grid_num_width=3)


    fig, ax = plt.subplots()
    ax.imshow(labels_out ,cmap=plt.get_cmap('Set3'),interpolation='nearest')
    plt.axis('off')

    height, width = labels_out.shape

    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.show()

