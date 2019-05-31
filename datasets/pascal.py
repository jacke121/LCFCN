import torch, os
import torch.utils.data as data

import numpy as np
import torchvision.transforms.functional as FT
from PIL import Image
import utils as ut

def get_pointDict(path, imgNames):
    pointDict = {}
    for i, img_name in enumerate(imgNames):
        print(i, img_name)
        pointDict[img_name] = []
        objects = ut.read_xml(path + "/Annotations/%s.xml" % img_name).findAll("object")
        for obj in objects:
            if int(obj.difficult.text):
                continue

            ymin = int(obj.ymin.text)
            ymax = int(obj.ymax.text)

            xmin = int(obj.xmin.text)
            xmax = int(obj.xmax.text)

            y = (ymax + ymin) // 2
            x = (xmax + xmin) // 2
            name = obj.find("name").text
            pointDict[img_name] += [{"y":y, "x":x,"name":name}]
      
            
    return pointDict

class Pascal2007(data.Dataset):
    name2class = {  
                   "aeroplane":0,
                   "bicycle":1,
                   "bird":2,
                   "boat":3,
                   "bottle":4,
                   "bus":5,
                   "car":6,
                   "cat":7,
                   "chair":8,
                   "cow":9,
                   "diningtable":10,
                   "dog":11,
                   "horse":12,
                   "motorbike":13,
                   "person":14,
                   "pottedplant":15,
                   "sheep":16,
                   "sofa":17,
                   "train":18,
                   "tvmonitor":19
                }

    def __init__(self,
                 split="val",
                 transform_function=None):
    
        self.path = path = r"E:\github\darknet_windows\build\darknet\x64\data\voc\VOCdevkit_voc\VOC2007/"
        self.transform_function = transform_function
        
        fname_path =  "%s/ImageSets/Main" % path
        path_pointJSON = "%s/pointDict.json" % path
       
        if split == "train":           
            self.imgNames = [t.replace("\n","") 
                                for t in 
                                ut.read_text(fname_path + "/train.txt")]

        elif split == "val":
            self.imgNames = [t.replace("\n","") 
                                for t in 
                                ut.read_text(fname_path + "/val.txt")]
        elif split == "test":
            self.imgNames = [t.replace("\n","") 
                                for t in 
                                ut.read_text(fname_path + "/val.txt")]
        
        if os.path.exists(path_pointJSON):
          self.pointsJSON = ut.load_json(path_pointJSON)
        else:
          pointDict = get_pointDict(path, self.imgNames)
          ut.save_json(path_pointJSON, pointDict)

        # for j, key in enumerate(pointDict):
        #   print(j)
        #   pList1 = pointDict[key]
        #   pList2 = self.pointsJSON[key]

        #   for p1 in pList1:
        #     y, x = p1["y"], p1["x"]
        #     flag = False
        #     for p2 in pList2:
        #       y2, x2 = p2["y"], p2["x"]
        #       if y == y2 and x == x2:
        #         flag = True
        #         break
        #     assert flag == True

        self.split = split
        self.n_classes = 21
            
        
    def __len__(self):
        return len(self.imgNames)

        
    def __getitem__(self, index):
        img_name =  self.imgNames[index]

        path2007 = self.path
        img_path = path2007 + "/JPEGImages/%s.jpg"% img_name
        img = Image.open(img_path).convert('RGB')
        
        # GET POINTS
        w, h = img.size
        points = np.zeros((h, w, 1))
        counts = np.zeros(20)
        counts_difficult = np.zeros(20)

        if self.split == "test":

            xml_path = path2007 + "/Annotations/%s.xml"% img_name
            xml = ut.read_xml(xml_path)
            for obj in xml.find_all("object"):
                if int(obj.find("difficult").text) == 1:
                    name = obj.find("name").text
                    counts_difficult[self.name2class[name]] += 1
                    continue

                name = obj.find("name").text
                xmin, xmax = obj.find("xmin").text, obj.find("xmax").text
                xmin, xmax = int(xmin), int(xmax)

                ymin, ymax = obj.find("ymin").text, obj.find("ymax").text
                ymin, ymax = int(ymin), int(ymax)

                yc = (ymax + ymin) // 2
                xc = (xmax + xmin) // 2

                points[yc, xc] = self.name2class[name] + 1
                counts[self.name2class[name]] += 1
                counts_difficult[self.name2class[name]] += 1

        else:
            pointLocs = self.pointsJSON[img_name]

            for p in pointLocs: 
                if  int(p["x"]) > w or int(p["y"]) > h:
                    continue
                else:
                    points[int(p["y"]), int(p["x"])] = p["cls"]
                    counts[p["cls"]-1] += 1
                    counts_difficult[p["cls"]-1] += 1

        points = FT.to_pil_image(points.astype("uint8"))
        if self.transform_function is not None:
            img, points = self.transform_function([img, points])

        return {"counts":torch.LongTensor(counts), 
                "images":img, "points":points, 
                "image_path":img_path,
                "index":index, "name":img_name}
