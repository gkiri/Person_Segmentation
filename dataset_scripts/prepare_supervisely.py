
import argparse
import base64
import json
import zlib
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    return cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=Path, help="Path to folder with images")
    parser.add_argument("-o", "--output_folder", type=Path, help="Path to the output folder")
    return parser.parse_args()

from pathlib import Path
Image_folder ='D:\Deep_Learning_Workspace\Project-Person Detection\data\supervisely\Supervisely Person Dataset (demo sample)\Supervisely Person Dataset (demo sample)'
Image_folder= Path(Image_folder)
Label_folder ='D:\Deep_Learning_Workspace\Project-Person Detection\data\supervisely\Supervisely Person Dataset (demo sample)\Supervisely Person Dataset (demo sample)\ds1\ann'
OImage_folder ='D:\Deep_Learning_Workspace\Project-Person Detection\data\supervisely\Final_Dataset\images'
OLabel_folder ='D:\Deep_Learning_Workspace\Project-Person Detection\data\supervisely\Final_Dataset\lables'
output_folder ='D:\Deep_Learning_Workspace\Project-Person Detection\data\supervisely\Final_Dataset'
output_folder = Path(output_folder)
def main():
    #args = get_args()
    output_image_folder = output_folder / "images"
    output_image_folder.mkdir(exist_ok=True, parents=True)

    output_label_folder = output_folder / "labels"
    output_label_folder.mkdir(exist_ok=True, parents=True)
    
    for folder in Image_folder.glob("*"):
        if folder.is_file():
            continue
        print("Gkiri::begin=",folder)
        for label_file_name in tqdm(sorted((folder / "ann").glob("*.json"))):
            with open(label_file_name) as f:
                ann = json.load(f)
                #print(ann)
            msk = np.zeros([ann["size"]["height"], ann["size"]["width"], 1], dtype=np.uint8)
            for person in ann["objects"]:
                if person.get("bitmap") and person.get("bitmap").get("data"):
                    z = zlib.decompress(base64.b64decode(person["bitmap"]["data"]))
                    n = np.fromstring(z, np.uint8)
                    pmsk = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(np.uint8)
                    y = person["bitmap"]["origin"][1]
                    x = person["bitmap"]["origin"][0]
                    msk[y : y + pmsk.shape[0], x : x + pmsk.shape[1], 0] += pmsk
                if person.get("points") and person.get("points").get("exterior"):
                    cv2.fillPoly(
                        msk,
                        [np.array(person["points"]["exterior"], dtype=np.int32)],
                        (255, 255, 255),
                    )
                if person.get("points") and person.get("points").get("interior"):
                    for inter in person["points"]["interior"]:
                        cv2.fillPoly(msk, [np.array(inter, dtype=np.int32)], (0, 0, 0))

            # print("Gkiri::",str(folder / "img" / label_file_name.stem))
            image = cv2.imread(str(folder / "img" / label_file_name.stem))
            if image is not None:
                cv2.imwrite(str(output_image_folder / f"{label_file_name.stem.split('.')[0]}.jpg"), image)
                cv2.imwrite(str(output_label_folder / f"{label_file_name.stem.split('.')[0]}.png"), msk[:, :, 0])

if __name__ == "__main__":
    main()