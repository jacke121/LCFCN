import cv2

import matplotlib

matplotlib.use('Agg')
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="7"

import argparse

import torch
import utils as ut
import torchvision.transforms.functional as FT
from skimage.io import imread, imsave
from torchvision import transforms
from models import model_dict


def apply(image_path, model_name, model_path):
    transformer = ut.ComposeJoint(
        [
            [transforms.ToTensor(), None],
            [transforms.Normalize(*ut.mean_std), None],
            [None, ut.ToLong()]
        ])

    # Load best model
    model = model_dict[model_name](n_classes=2).cuda()
    model.load_state_dict(torch.load(model_path))

    # Read Image
    image_raw = imread(image_path)
    cv2.imshow("img",image_raw)
    cv2.waitKeyEx()
    collection = list(map(FT.to_pil_image, [image_raw, image_raw]))
    image, _ = transformer(collection)

    batch = {"images": image[None]}

    # Make predictions
    pred_blobs = model.predict(batch, method="blobs").squeeze()
    pred_counts = int(model.predict(batch, method="counts").ravel()[0])

    # Save Output
    save_path = "figures/_blobs_count_{}.png".format(pred_counts)

    imsave(save_path, ut.combine_image_blobs(image_raw, pred_blobs))
    print("| Counts: {}\n| Output saved in: {}".format(pred_counts, save_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-image_path', '--image_path', default="figures/1.jpg")
    parser.add_argument('-model_path', '--model_path', default="best_model_coco.pth")
    parser.add_argument('-model_name', '--model_name', default="ResFCN")

    args = parser.parse_args()

    # dataset_name, model_name, metric_name = experiments.get_experiment(args.exp_name)

    apply(args.image_path, args.model_name, args.model_path)



if __name__ == "__main__":
    main()
