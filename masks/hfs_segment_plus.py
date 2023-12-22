import matplotlib.pyplot as plt 
from PIL import Image
from skimage import io
import cv2
import numpy as np
import cc3d
import imageio


def find_neibor(res):
    sobelxy1 = cv2.Sobel(src=res, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    sobelxy2 = cv2.Sobel(src=res, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobelxy = cv2.addWeighted(sobelxy1, 0.5, sobelxy2, 0.5, 0)
    whole_dict = {}
    for i in range(1, np.max(res) + 1):
        whole_dict[i] = []
    for i in range(1,sobelxy.shape[0]-1):
        for j in range(1,sobelxy.shape[1]-1):
            if sobelxy[i,j] != 0:
                # local_res = res[i-1:i+2,j-1:j+2]
                dict2 = np.unique(res[i-1:i+2,j-1:j+2],return_counts=True)
                keys = dict2[0]
                for i_i in range(keys.shape[0]):
                    for j_j in range(keys.shape[0]):
                        if i_i != j_j and keys[j_j] not in whole_dict[keys[i_i]]:
                            whole_dict[keys[i_i]].append(keys[j_j])

    # left boundary
    for i in range(1,sobelxy.shape[0]-1):
        if sobelxy[i, 0] != 0:
            # local_res = res[i-1:i+2,j-1:j+2]
            dict2 = np.unique(res[i - 1:i + 2, 0: 2], return_counts=True)
            keys = dict2[0]
            for i_i in range(keys.shape[0]):
                for j_j in range(keys.shape[0]):
                    if i_i != j_j and keys[j_j] not in whole_dict[keys[i_i]]:
                        whole_dict[keys[i_i]].append(keys[j_j])

    # right boundary
    for i in range(1,sobelxy.shape[0]-1):
        if sobelxy[i, sobelxy.shape[1]-1] != 0:
            # local_res = res[i-1:i+2,j-1:j+2]
            dict2 = np.unique(res[i - 1:i + 2, sobelxy.shape[1]-2:sobelxy.shape[1]], return_counts=True)
            keys = dict2[0]
            for i_i in range(keys.shape[0]):
                for j_j in range(keys.shape[0]):
                    if i_i != j_j and keys[j_j] not in whole_dict[keys[i_i]]:
                        whole_dict[keys[i_i]].append(keys[j_j])

    # upper boundary
    for j in range(1,sobelxy.shape[1]-1):
        if sobelxy[0, j] != 0:
            # local_res = res[i-1:i+2,j-1:j+2]
            dict2 = np.unique(res[0:2, j - 1:j + 2], return_counts=True)
            keys = dict2[0]
            for i_i in range(keys.shape[0]):
                for j_j in range(keys.shape[0]):
                    if i_i != j_j and keys[j_j] not in whole_dict[keys[i_i]]:
                        whole_dict[keys[i_i]].append(keys[j_j])

    # lower boundary
    for j in range(1,sobelxy.shape[1]-1):
        if sobelxy[sobelxy.shape[0]-1, j] != 0:
            # local_res = res[i-1:i+2,j-1:j+2]
            dict2 = np.unique(res[sobelxy.shape[0]-2:sobelxy.shape[0], j - 1:j + 2], return_counts=True)
            keys = dict2[0]
            for i_i in range(keys.shape[0]):
                for j_j in range(keys.shape[0]):
                    if i_i != j_j and keys[j_j] not in whole_dict[keys[i_i]]:
                        whole_dict[keys[i_i]].append(keys[j_j])

    return whole_dict



def merge_smallest_label(labels_out, areas_dict, neibor_dict):
    labels = np.array(list(areas_dict.keys()))
    areas = np.array(list(areas_dict.values()))
    smallest_label = labels[np.argmin(areas)]
    # print('s', smallest_label)

    neibor_areas = [areas_dict[i] for i in neibor_dict[smallest_label]]
    # print()
    smallest_neibor_label = neibor_dict[smallest_label][np.argmin(neibor_areas)]

    # merge smallest_label to its smallest_neibor
    labels_out[labels_out == smallest_label] = smallest_neibor_label
    # labels2, areas2 = np.unique(labels_out, return_counts=True)

    # update areas_count and neibor_dict
    areas_dict[smallest_neibor_label] += areas_dict[smallest_label]
    areas_dict[smallest_label] = (labels_out.shape[0] * labels_out.shape[1])+1

    neibor_labels = neibor_dict[smallest_label]

    for neibor_label in neibor_labels:
        neibor_dict[neibor_label].remove(smallest_label)
        if neibor_label == smallest_neibor_label:
            for neibor_label_i in neibor_labels:
                if neibor_label_i != neibor_label and neibor_label_i not in neibor_dict[neibor_label]:
                    neibor_dict[neibor_label].append(neibor_label_i) 
        elif neibor_label != smallest_neibor_label:
            if smallest_neibor_label not in neibor_dict[neibor_label]:
                neibor_dict[neibor_label].append(smallest_neibor_label)

    neibor_dict.pop(smallest_label)
    return labels_out, areas_dict, neibor_dict



def find_neibor2(labels_out):
    labels_out_cc_expand = np.expand_dims(labels_out, axis=2)
    surface_per_contact = cc3d.contacts(labels_out_cc_expand, connectivity=26, surface_area=False)
    dict_0 = {}

    for i in range(1, np.max(labels_out_cc_expand) + 1):
        dict_0[i] = []
    for key in surface_per_contact.keys():
        key_a, key_b = key
        dict_0[key_a].append(key_b)
        dict_0[key_b].append(key_a)

    return dict_0


def merge_label(labels_out, patch_num = 4):
    labels, areas = np.unique(labels_out,return_counts=True)
    areas_dict = {}
    for i in range(len(labels)):
        areas_dict[labels[i]] = areas[i]

    neibor_dict = find_neibor2(labels_out)

    while len(neibor_dict) > patch_num:
        # print(len(neibor_dict))
        labels_out, areas_dict, neibor_dict = merge_smallest_label(labels_out, areas_dict, neibor_dict)
        # plt.imshow(labels_out)
        # plt.show()

    # re_label
    labels_final = np.unique(labels_out, return_counts=False)

    for i in range(labels_final.shape[0]):
        labels_out[labels_out == labels_final[i]] = i


    return labels_out


def hfs_domain_decompostion(img,patch_num=4):
    # plt.imshow(img)
    # plt.show()
    # print(img.shape)
    engine = cv2.hfs.HfsSegment_create(img.shape[0], img.shape[1])

    res = engine.performSegmentGpu(img, False)
    # print('heads:', np.max(res))
    # plt.imshow(res)
    # plt.show()
    # return res
    # print('s')
    labels_out_cc = cc3d.connected_components(res, connectivity=8)

    # graph = cc3d.voxel_connectivity_graph(labels_out_cc, connectivity=8)

    # print('heads2:', np.max(labels_out_cc))
    # plt.imshow(labels_out_cc)
    # plt.show()

    labels_out = merge_label(labels_out_cc, patch_num=patch_num)
    # plt.imshow(labels_out)
    # plt.show()
    # return res
    return labels_out








if __name__ == '__main__':
    img = cv2.imread('../data/001_L.png')

    plt.imshow(img)
    plt.show()
    labels_out = hfs_domain_decompostion(img,patch_num=9)

    plt.imshow(labels_out)
    plt.show()


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






