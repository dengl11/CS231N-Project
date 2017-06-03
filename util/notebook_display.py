import numpy as np
import matplotlib.pyplot as plt

def disable_axis(ax):
	ax.set_xticklabels([])
	ax.set_yticklabels([])

def display_gif(path, name, width=300):
    return display.HTML('<h3>{}</h3> <img src="{}", width={}>'.format(name, path, width))


def sample_img(imgs):
    return imgs[np.random.choice(range(len(imgs)))]

def sample_img_many(imgs, num, inorder=True):
    """
    return a sample of images from a list
    Input:
        imgs:       [img] or nd array
        inorder:    order of images sampled is sorted
    """
    if inorder: return imgs[sorted(np.random.choice(range(len(imgs)), num, replace=False))]
    return imgs[np.random.choice(range(len(imgs)), num, replace=False)]


def sample_and_show(imgs, size = (12, 6)):
    sample = sample_img(imgs)
    plot_img(sample, size = size)
    return sample

def sample_pred_frames(imgs, gap = 3, num = 6, inorder=True):
    n = imgs.shape[0]
    before = np.random.choice(range(n-gap-1), num, replace=False)
    if inorder: before.sort()
    after = before + gap +1
    mid = (before + after)//2
    return (imgs[before], imgs[after], imgs[mid])


def plot_img(img, size = None, ax=None):
    max_val = np.max(img)
    if size: plt.figure(figsize=size)
    if 'float' in str(img.dtype) and max_val < 10:
        plt.imshow(convert_for_display(img))
    else:
        plt.imshow(img)
    if ax: disable_axis(ax)


def convert_for_display(img_batch):
    """
    [-1, 1] -> [0, 255]
    """
    imgs = (img_batch+1)/2 * 255
    return imgs.astype('uint8')


def plot_images(imgs, size = (12, 6), title=None, sub_titles=[]):
    """
    Input: 
        imgs: [image]
    """
    fig = plt.figure(figsize=size)
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(1, len(imgs), i+1)
        if img.shape[-1] == 1: img = img.reshape(img.shape[:-1]) # in case of one channel
        if sub_titles: plt.title(sub_titles[i])
        plot_img(img)
        disable_axis(ax)
    if title: plt.suptitle(title)
    plt.show()



def plot_images_ndarray(imgs, size = (12, 6), title=None, sub_titles=[]):
    """
    Input: 
        imgs: nd array
    """
    plot_images(list(imgs), size = size, title=title, sub_titles=sub_titles)



def plot_batch_images(imgs, size = (12, 6), title=None):
    """
    Input: 
        imgs: nd array of [batch_size, ...]
    """
    plot_images(list(imgs), size, title)


def sample_and_show_many(imgs, num, size = (12, 6), inorder=True):
    samples = sample_img_many(imgs, num, inorder = inorder)
    plot_images(samples, size = size)
    return samples
    