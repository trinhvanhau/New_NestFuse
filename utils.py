import os
import random
import numpy as np
import torch
from PIL import Image
from args_fusion import args
from imageio import imread, imwrite
from skimage.transform import resize
import matplotlib as mpl
from os import listdir
from os.path import join


def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * img.size[1] / img.size[0])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=False):
    if cuda:
        img = tensor.cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def matSqrt(x):
    U, D, V = torch.svd(x)
    return U @ torch.diag(D.pow(0.5)) @ V.t()


def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print(f'BATCH SIZE {BATCH_SIZE}.')
    print(f'Train images number {num_imgs}.')
    print(f'Train images samples {num_imgs / BATCH_SIZE}.')

    if mod > 0:
        print(f'Train set has been trimmed {mod} samples...\n')
        original_imgs_path = original_imgs_path[:-mod]
    batches = len(original_imgs_path) // BATCH_SIZE
    return original_imgs_path, batches


def get_image(path, height=256, width=256, flag=False):
    image = imread(path)
    if not flag:  # Grayscale
        image = np.mean(image, axis=-1).astype(image.dtype)
    if height is not None and width is not None:
        image = resize(image, (height, width), preserve_range=True).astype(image.dtype)
    return image


def get_test_image(paths, height=None, width=None, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, flag)
        base_size = 512
        h, w = image.shape[:2]
        c = 1
        if h > base_size or w > base_size:
            c = 4
            images = get_img_parts(image, h, w)
        else:
            image = np.reshape(image, [1, h, w])
            images.append(image)
            images = np.stack(images, axis=0)
            images = torch.from_numpy(images).float()
    return images, h, w, c


def get_img_parts(image, h, w):
    images = []
    h_cen = h // 2
    w_cen = w // 2
    img1 = image[0:h_cen + 3, 0:w_cen + 3]
    img2 = image[0:h_cen + 3, w_cen - 2:w]
    img3 = image[h_cen - 2:h, 0:w_cen + 3]
    img4 = image[h_cen - 2:h, w_cen - 2:w]
    for img in [img1, img2, img3, img4]:
        img = np.reshape(img, [1, 1, img.shape[0], img.shape[1]])
        images.append(torch.from_numpy(img).float())
    return images


def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    if args.cuda:
        img_fusion = img_fusion.cpu().numpy()
    else:
        img_fusion = img_fusion.numpy()
    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = (img_fusion * 255).astype('uint8').transpose(1, 2, 0)
    img = Image.fromarray(img_fusion)
    img.save(output_path)


def get_train_images(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images_ir, images_vi = [], []
    for path in paths:
        images_ir.append(get_image(path, height, width, flag))
        images_vi.append(get_image(path.replace('lwir', 'visible'), height, width, flag))
    return torch.from_numpy(np.stack(images_ir)).float(), torch.from_numpy(np.stack(images_vi)).float()

def get_train_images_auto(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, flag)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, height, width])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images
    
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list(
        'cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00', '#FF0000', '#8B0000'], 256
    )
