from pathlib import Path
from glob import glob
import os
import json
from tqdm import tqdm
from ultralytics import YOLO
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.iterators import create_patch_iterator, PatchConfiguration
from wholeslidedata.annotation.labels import Label
import torch
from structures import Point
from ensemble_boxes import *
from torchvision import transforms



INPUT_PATH = Path("/input")
EVALUATION_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")


def run():
    # Collect all .tif files
    image_paths = sorted(glob(os.path.join(INPUT_PATH, "images/kidney-transplant-biopsy-wsi-pas/*.tif")))
    mask_paths = sorted(glob(os.path.join(INPUT_PATH, "images/tissue-mask/*.tif")))

    output_metadata = []

    for image_path, mask_path in zip(image_paths, mask_paths):
        # Infer output filenames from image filename
        base_name = Path(image_path).stem
        case_output_path = EVALUATION_PATH / "input" / base_name / "output"
        case_output_path.mkdir(parents=True, exist_ok=True)

        json_filename_lymphocytes = case_output_path / "detected-lymphocytes.json"
        json_filename_monocytes = case_output_path / "detected-monocytes.json"
        json_filename_inflammatory_cells = case_output_path / "detected-inflammatory-cells.json"

        weight_root = os.path.join(MODEL_PATH, "bestPAS256.pt")
        weight_root2 = os.path.join(MODEL_PATH, "bestPAS.pt")
        
        patch_shape = (256, 256, 3)
        spacings = (0.5,)
        overlap = (0, 0)
        offset = (0, 0)
        center = False

        patch_configuration = PatchConfiguration(
            patch_shape=patch_shape,
            spacings=spacings,
            overlap=overlap,
            offset=offset,
            center=center
        )

        YOLOmodel = YOLO(weight_root)
        YOLOmodel = torch.compile(YOLOmodel)
        #print(torch.__version__)
        YOLOmodel2 = YOLO(weight_root2)
        
        iterator = create_patch_iterator(
            image_path=image_path,
            mask_path=mask_path,
            patch_configuration=patch_configuration,
            cpus=4,
            backend='asap'
        )
        
        # Wait until the iterator is ready by accessing the first element
        try:
            first_patch = next(iterator)
            print("Iterator loaded successfully.")
        except StopIteration:
            print("Iterator is empty.")
        except Exception as e:
            print(f"Error while waiting for iterator: {e}")

        inference(
            iterator=iterator,
            predictor=[YOLOmodel,YOLOmodel2],
            spacing=spacings[0],
            image_path=image_path,
            output_path=case_output_path,
            json_filename_lymphocytes=json_filename_lymphocytes,
            json_filename_monocytes=json_filename_monocytes,
            json_filename_inflammatory_cells=json_filename_inflammatory_cells,
        )

        iterator.stop()

        # Add metadata for this image
        output_metadata.append({
            "pk": base_name,
            "inputs": [
                {
                    "image": {
                        "name": Path(image_path).name
                    },
                    "interface": {
                        "slug": "kidney-transplant-biopsy",
                        "kind": "Image",
                        "super_kind": "Image",
                        "relative_path": "images/kidney-transplant-biopsy-wsi-pas"
                    }
                }
            ],
            "outputs": [
                {
                    "interface": {
                        "slug": "detected-lymphocytes",
                        "kind": "Multiple points",
                        "super_kind": "File",
                        "relative_path": "detected-lymphocytes.json"
                    }
                },
                {
                    "interface": {
                        "slug": "detected-monocytes",
                        "kind": "Multiple points",
                        "super_kind": "File",
                        "relative_path": "detected-monocytes.json"
                    }
                },
                {
                    "interface": {
                        "slug": "detected-inflammatory-cells",
                        "kind": "Multiple points",
                        "super_kind": "File",
                        "relative_path": "detected-inflammatory-cells.json"
                    }
                }
            ]
        })

    # Save the combined metadata JSON
    metadata_file = EVALUATION_PATH / "input" / "predictions.json"
    write_json_file(location=metadata_file, content=output_metadata)
    print(f"Metadata JSON saved to {metadata_file}")


# Inference function
# (Same as before, adjusted for multiple filenames)
def inference(iterator, predictor, spacing, image_path, output_path, json_filename_lymphocytes, json_filename_monocytes, json_filename_inflammatory_cells):
    print("predicting... ", image_path)
    output_dict_lymphocytes = {
        "name": "lymphocytes",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    output_dict_monocytes = {
        "name": "monocytes",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    output_dict_inflammatory_cells = {
        "name": "inflammatory-cells",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }
    
    target_mean = [183.68832/255.0, 162.18071/255.0, 186.88966/255.0] 
    target_std = [57.35563/255.0,  68.62531/255.0,  54.158657/255.0] 
    normalize_transform = transforms.Normalize(mean=target_mean, std=target_std)

    spacing_min = 0.25
    ratio = spacing / spacing_min
    
    with WholeSlideImage(image_path) as wsi:
        spacing = wsi.get_real_spacing(spacing_min)
        print(spacing)

    pcount = 0
    for x_batch, y_batch, info in iterator:
        x_batch = x_batch.squeeze(0)
        y_batch = y_batch.squeeze(0)
        c = info['x'] 
        r = info['y'] 
        pcount = pcount + 1
        normalized_img_tensor = normalize_transform(torch.from_numpy(x_batch).permute(0, 3, 1, 2).float() / 255.0)
        image = normalized_img_tensor #torch.from_numpy(x_batch).permute(0, 3, 1, 2).float() / 255.0
        image = torch.from_numpy(x_batch).permute(0, 3, 1, 2).float() / 255.0
        #results = predictor.predict(image, imgsz=512, max_det=1000, conf=0.4, iou=0.7, augment=True)#augment=True, agnostic_nms=True
        results = predictor[0].predict(image, imgsz=256, max_det=1000, conf=0.4, iou=0.7, augment=True)#augment=True, agnostic_nms=True
        results2 = predictor[1].predict(image, imgsz=256, max_det=1000, conf=0.25, iou=0.75, augment=True)#augment=True, agnostic_nms=True
        
        #https://github.com/ZFTurbo/Weighted-Boxes-Fusion
        boxes0 = []
        boxes1 = []
        scores0 = []
        scores1 = []
        labels0 = [] 
        labels1 = []
        for idx,result in enumerate(results):
            boxes = result.boxes
            for box in boxes:
                boxes0.append(box.data.cpu().numpy()[:, :4][0]/256.0)
                scores0.append(float(box.conf.cpu().numpy()[0]))
                labels0.append(int(box.cls.cpu().numpy()[0]))
        for idx,result in enumerate(results2):
            boxes = result.boxes
            for box in boxes:
                boxes1.append(box.data.cpu().numpy()[:, :4][0]/256.0)
                scores1.append(float(box.conf.cpu().numpy()[0]))
                labels1.append(int(box.cls.cpu().numpy()[0]))
        
        boxes_list = [boxes0, boxes1] #... add your boxes
        labels_list = [labels0, labels1] #...
        scores_list = [scores0, scores1] #...
        weights = [1, 1]
        iou_thr = 0.5
        sigma = 0.1
        skip_box_thr = 0.0001
        
        boxes_fused, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        
        for box, cls, prob in zip(boxes_fused,labels,scores):
            box = box*256
            xmin, ymin, xmax, ymax = box
        
            x_center = ((xmin + xmax) / 2.0)* ratio + c# * ratio + c
            y_center = ((ymin + ymax) / 2.0)* ratio + r# * ratio + r
            #x_center = x_center * 0.24199951445730394*2
            #y_center = y_center * 0.24199951445730394*2
            #cls = int(box.cls.cpu().numpy()[0])
            #prob = float(box.conf.cpu().numpy()[0])
            #if x_center == 255 or y_center == 255:
            #    continue

            point_record = {
                "name": f"Point {len(output_dict_lymphocytes['points'])}",
                "point": [px_to_mm(x_center, spacing), px_to_mm(y_center, spacing), spacing],#0.24199951445730394],
                "probability": prob
            }

            if cls == 0:
                output_dict_lymphocytes["points"].append(point_record)
            elif cls == 1:
                output_dict_monocytes["points"].append(point_record)

            output_dict_inflammatory_cells["points"].append(point_record)
        '''
        for idx,result in enumerate(results):
            boxes = result.boxes
            for box in boxes:
                bbox = box.data.cpu().numpy()
                xmin, ymin, xmax, ymax = bbox[:, :4][0]
                #TODO: here is something wrong
                x_center = ((xmin + xmax) / 2.0)* ratio + c# * ratio + c
                y_center = ((ymin + ymax) / 2.0)* ratio + r# * ratio + r
                #x_center = x_center * 0.24199951445730394*2
                #y_center = y_center * 0.24199951445730394*2
                cls = int(box.cls.cpu().numpy()[0])
                prob = float(box.conf.cpu().numpy()[0])
                #if x_center == 255 or y_center == 255:
                #    continue

                point_record = {
                    "name": f"Point {len(output_dict_lymphocytes['points'])}",
                    "point": [px_to_mm(x_center, spacing), px_to_mm(y_center, spacing), 0.24199951445730394],
                    "probability": prob
                }

                if cls == 0:
                    output_dict_lymphocytes["points"].append(point_record)
                elif cls == 1:
                    output_dict_monocytes["points"].append(point_record)

                output_dict_inflammatory_cells["points"].append(point_record)
            '''
    print(f"Predicted {len(output_dict_lymphocytes['points'])} lymphocytes")
    print(f"Predicted {len(output_dict_monocytes['points'])} monocyte")
    print(f"Predicted {len(output_dict_inflammatory_cells['points'])} inflammatory_cells")
    print(pcount)
    print("saving predictions...")
    
    # Save JSON files
    write_json_file(location=json_filename_lymphocytes, content=output_dict_lymphocytes)
    write_json_file(location=json_filename_monocytes, content=output_dict_monocytes)
    write_json_file(location=json_filename_inflammatory_cells, content=output_dict_inflammatory_cells)
    
    json_filename_lymphocytes = EVALUATION_PATH / "detected-lymphocytes.json"
    json_filename_monocytes = EVALUATION_PATH / "detected-monocytes.json"
    json_filename_inflammatory_cells = EVALUATION_PATH / "detected-inflammatory-cells.json"
    write_json_file(location=json_filename_lymphocytes, content=output_dict_lymphocytes)
    write_json_file(location=json_filename_monocytes, content=output_dict_monocytes)
    write_json_file(location=json_filename_inflammatory_cells, content=output_dict_inflammatory_cells)     
        
    print("Predictions saved.")


def write_json_file(location, content):
    with open(location, 'w') as f:
        json.dump(content, f, indent=4)


def px_to_mm(px, spacing):
    return px * spacing / 1000


if __name__ == "__main__":
    run()
