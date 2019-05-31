import torch
import utils as ut
import pandas as pd 

from torchvision import transforms
from datasets import dataset_dict
from models import model_dict

def test(dataset_name="pascal", model_name="ResFCN", metric_name="mRMSE",
         path_history="/", path_best_model="best_model.pth"):

  history = ut.load_json("history.json")

  transformer = ut.ComposeJoint(
                    [
                         [transforms.ToTensor(), None],
                         [transforms.Normalize(*ut.mean_std), None],
                         [None,  ut.ToLong() ]
                    ])  
  test_set = dataset_dict[dataset_name](split="test", 
                                       transform_function=transformer)

  model = model_dict[model_name](n_classes=test_set.n_classes).cuda()
  # path_best_model = "/mnt/home/issam/LCFCNSaves/pascal/State_Dicts/best_model.pth"
  model.load_state_dict(torch.load(path_best_model))

  # model.trained_images = set(history["trained_images"])

  testDict = ut.val(model=model, dataset=test_set, 
                    epoch=history["best_val_epoch"], 
                    metric_name=metric_name)

  print(pd.DataFrame([testDict]))

if __name__ == '__main__':
    test()
