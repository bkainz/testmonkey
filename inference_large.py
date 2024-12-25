import os
import json
from glob import glob
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import shutil

import sys
current_script_path = os.path.dirname(os.path.abspath(__file__))
if current_script_path not in sys.path:
    sys.path.append(current_script_path)
  
sys.path.append("/opt/app/biomp")
sys.path.append("/opt/app/biomp/modeling")
sys.path.append("/opt/app/biomp/utilities")
sys.path.append("/opt/app/biomp/inference_utils")
sys.path.append("/opt/app/")

os.environ["HF_HOME"] = "/opt/app/cache/huggingface"
os.environ["HUGGINGFACE_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from ultralytics import YOLO
from tqdm import tqdm
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.iterators import create_patch_iterator, PatchConfiguration
from torchvision import transforms
from ensemble_boxes import weighted_boxes_fusion

import biomp
import biomp.utilities
from biomp.modeling.BaseModel import BaseModel
from biomp.modeling import build_model
from biomp.utilities.distributed import init_distributed
from biomp.utilities.arguments import load_opt_from_config_files
from biomp.utilities.constants import BIOMED_CLASSES

from biomp.inference_utils.inference import interactive_infer_image
from biomp.inference_utils.output_processing import check_mask_stats
from biomp.modeling import build_model
#from utilities.distributed import init_distributed
from biomp.utilities.arguments import load_opt_from_config_files
from biomp.utilities.constants import BIOMED_CLASSES
from biomp.inference_utils.inference import interactive_infer_image
from biomp.inference_utils.output_processing import check_mask_stats
import numpy as np
from PIL import Image

import multiprocessing
pool = multiprocessing.Pool(processes=1)

BPARSET = 0.25
YOLOT = 0.25

# -------------------------------------------------------------------------
# Global Paths
# -------------------------------------------------------------------------
INPUT_PATH = Path("/input")
EVALUATION_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")

# -------------------------------------------------------------------------
# Main Execution Entry Point
# -------------------------------------------------------------------------
def run():
    """
    Main function that:
    1. Gathers all .tif files from the input.
    2. Creates patch iterators.
    3. Runs inference + Weighted Boxes Fusion for each image.
    4. Saves the consolidated JSON metadata to 'predictions.json'.
    """
    # Collect .tif files from each folder
    image_paths = sorted(glob(os.path.join(INPUT_PATH, "images/kidney-transplant-biopsy-wsi-pas/*.tif")))
    mask_paths = sorted(glob(os.path.join(INPUT_PATH, "images/tissue-mask/*.tif")))

    # Prepare metadata for all images
    output_metadata = []

    # Model weights (you can add as many as you like here)
    weights = [
        "bestPAS256.pt",
        "best_PASIHC256.pt",#"bestPAS.pt",
        "best2k_PAS256.pt"
    ]
    # Construct absolute paths to these weights
    weight_paths = [os.path.join(MODEL_PATH, w) for w in weights]

    # Load and compile all YOLO models
    print("Loading models...")
    predictors = []
    for wp in weight_paths:
        model = YOLO(wp)
        model = torch.compile(model)  # optional speedup
        predictors.append(model)
    print(f"Loaded {len(predictors)} YOLO models.")

    opt = load_opt_from_config_files(["/opt/app/biomp/configs/biomedparse_inference.yaml"])
    opt = init_distributed(opt)
    
    pretrained_pth =  MODEL_PATH / "model_state_dict.pt"
    model = torch.compile(BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth.absolute().as_posix()).eval().cuda())
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)
            
    # Iterate over each (image, mask) pair
    for image_path, mask_path in zip(image_paths, mask_paths):
        base_name = Path(image_path).stem

        # Output folder for this case
        case_output_path = EVALUATION_PATH / "input" / base_name / "output"
        case_output_path.mkdir(parents=True, exist_ok=True)

        # Define JSON output file names
        json_filename_lymphocytes256 = case_output_path / "detected-lymphocytes256.json"
        json_filename_monocytes256 = case_output_path / "detected-monocytes256.json"
        json_filename_inflammatory_cells256 = case_output_path / "detected-inflammatory-cells256.json"
        
        json_filename_lymphocytes1024 = case_output_path / "detected-lymphocytes1024.json"
        json_filename_monocytes1024 = case_output_path / "detected-monocytes1024.json"
        json_filename_inflammatory_cells1024 = case_output_path / "detected-inflammatory-cells1024.json"

        # Configure patch extraction
        patch_configuration = PatchConfiguration(
            patch_shape=(1024, 1024, 3),
            spacings=(0.5,),
            overlap=(0, 0),
            offset=(0, 0),
            center=False
        )

        # Create patch iterator
        iterator = create_patch_iterator(
            image_path=image_path,
            mask_path=mask_path,
            patch_configuration=patch_configuration,
            cpus=1,
            backend='asap'
        )
        # Confirm the iterator is ready
        try:
            img, _, _ = next(iterator)
            print(f"Iterator for {base_name} loaded successfully.")
        except StopIteration:
            print(f"Iterator for {base_name} is empty.")
        except Exception as e:
            print(f"Error initializing iterator for {base_name}: {e}")
            
        #pred_mask = interactive_infer_image(model, Image.fromarray(img[0,0,:,:,:]), ['lymphocytes', 'monocytes', 'inflammatory cells'])
        #print(pred_mask)
        
        inference_with_biomed_model(
            iterator=iterator,
            model=model,
            spacing=patch_configuration.spacings[0],
            image_path=image_path,
            output_path=case_output_path,
            json_filename_lymphocytes=json_filename_lymphocytes1024,
            json_filename_monocytes=json_filename_monocytes1024,
            json_filename_inflammatory_cells=json_filename_inflammatory_cells1024
        )

        # Stop the iterator once done
        iterator.stop()
        del iterator
        
        # Configure patch extraction
        patch_configuration = PatchConfiguration(
            patch_shape=(256, 256, 3),
            spacings=(0.5,),
            overlap=(0, 0),
            offset=(0, 0),
            center=False
        )

        # Create patch iterator
        iterator = create_patch_iterator(
            image_path=image_path,
            mask_path=mask_path,
            patch_configuration=patch_configuration,
            cpus=1,
            backend='asap'
        )
        
        # Run inference and save predictions
        
        inference(
            iterator=iterator,
            predictors=predictors,
            spacing=patch_configuration.spacings[0],
            image_path=image_path,
            output_path=case_output_path,
            json_filename_lymphocytes=json_filename_lymphocytes256,
            json_filename_monocytes=json_filename_monocytes256,
            json_filename_inflammatory_cells=json_filename_inflammatory_cells256
        )
        
        
        # Stop the iterator once done
        iterator.stop()
        del iterator
        
        top_level_lymph = EVALUATION_PATH / "detected-lymphocytes.json"
        top_level_mono = EVALUATION_PATH / "detected-monocytes.json"
        top_level_inflam = EVALUATION_PATH / "detected-inflammatory-cells.json"
        
        json_filename_lymphocytes = case_output_path / "detected-lymphocytes.json"
        json_filename_monocytes = case_output_path / "detected-monocytes.json"
        json_filename_inflammatory_cells = case_output_path / "detected-inflammatory-cells.json"
        
        fuse_points_with_confidence(json_filename_lymphocytes256,json_filename_lymphocytes1024,top_level_lymph)
        fuse_points_with_confidence(json_filename_lymphocytes256,json_filename_lymphocytes1024,json_filename_lymphocytes)
        fuse_points_with_confidence(json_filename_monocytes256,json_filename_monocytes1024,top_level_mono)
        fuse_points_with_confidence(json_filename_monocytes256,json_filename_monocytes1024,json_filename_monocytes)
        fuse_points_with_confidence(json_filename_inflammatory_cells256,json_filename_inflammatory_cells1024,top_level_inflam)
        fuse_points_with_confidence(json_filename_inflammatory_cells256,json_filename_inflammatory_cells1024,json_filename_inflammatory_cells)
    
        # Prepare metadata for this image
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

    del model
    #shutil.rmtree("./BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    
    # Save the combined metadata JSON
    metadata_file = EVALUATION_PATH / "input" / "predictions.json"
    write_json_file(location=metadata_file, content=output_metadata)
    print(f"Metadata JSON saved to {metadata_file}")
    
import json
import math
import os
from pathlib import Path

def fuse_points_with_confidence(
    json_a_path: str,
    json_b_path: str,
    output_json_path: str,
    distance_threshold: float = 0.005,
    weight_a: float = 1.0,
    weight_b: float = 1.0
):
    """
    Fuses points from two JSON files A and B, each containing:
      {
        "points": [
          {
            "name": "Point 0",
            "point": [x_coord, y_coord, spacing],
            "probability": float
          },
          ...
        ]
      }

    For each point in A:
      - If there's a point in B within 'distance_threshold', fuse them into one.
        * Weighted location and probability => then shift prob closer to 1.
      - If no match is found, keep the A point with half its probability.
    Leftover B points also remain, halved probability.

    :param json_a_path: Path to JSON A.
    :param json_b_path: Path to JSON B.
    :param output_json_path: Output path for the fused JSON.
    :param distance_threshold: Max distance in x,y to consider points matched.
    :param weight_a: Weight factor for A's probability.
    :param weight_b: Weight factor for B's probability.
    """

    def load_json_points(file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get("name", ""), data.get("points", [])

    def euclidean_distance_2d(pt_a, pt_b):
        """
        Each 'point' is [x, y, spacing]. We only compute the distance on (x,y).
        """
        return math.sqrt((pt_a[0] - pt_b[0])**2 + (pt_a[1] - pt_b[1])**2)

    # --- 1) Load from A and B
    name_a, points_a = load_json_points(json_a_path)
    name_b, points_b = load_json_points(json_b_path)

    fused_points = []
    used_b_indices = set()

    # --- 2) Match each point in A
    for a_point in points_a:
        a_xy = a_point["point"][:2]
        a_prob = a_point["probability"]
        a_spacing = a_point["point"][2]

        best_idx = None
        best_dist = float('inf')

        # find closest B within distance_threshold
        for b_idx, b_point in enumerate(points_b):
            if b_idx in used_b_indices:
                continue
            b_xy = b_point["point"][:2]
            dist = euclidean_distance_2d(a_xy, b_xy)
            if dist < distance_threshold and dist < best_dist:
                best_dist = dist
                best_idx = b_idx

        if best_idx is not None:
            # We found a match in B
            print("We found a match in B", a_xy, " ", b_xy)
            used_b_indices.add(best_idx)
            b_point = points_b[best_idx]
            b_xy = b_point["point"][:2]
            b_prob = b_point["probability"]
            b_spacing = b_point["point"][2]

            # Weighted location
            wa = weight_a * a_prob
            wb = weight_b * b_prob
            total_w = wa + wb
            if total_w > 0:
                fused_x = (wa * a_xy[0] + wb * b_xy[0]) / total_w
                fused_y = (wa * a_xy[1] + wb * b_xy[1]) / total_w
            else:
                fused_x, fused_y = a_xy

            # Weighted probability => then push closer to 1
            fused_prob = (wa + wb) / (weight_a + weight_b)
            # Shift prob half-way toward 1 => fused_prob + (1 - fused_prob)*0.5
            fused_prob = fused_prob + (1.0 - fused_prob)*0.5
            # clamp at 1
            if fused_prob > 1.0:
                fused_prob = 1.0

            fused_record = {
                "name": f"Fused {len(fused_points)}",
                "point": [fused_x, fused_y, a_spacing],  # or average A/B spacing
                "probability": fused_prob
            }
            fused_points.append(fused_record)
        else:
            # No match => keep A but half probability
            half_conf_record = {
                "name": f"NoMatchA {len(fused_points)}",
                "point": [a_xy[0], a_xy[1], a_spacing],
                "probability": a_prob * 0.5
            }
            #if "monocytes" in json_b_path.as_posix():
            fused_points.append(half_conf_record)

    # --- 3) Any leftover B points => also halved
    for b_idx, b_point in enumerate(points_b):
        if b_idx not in used_b_indices:
            b_xy = b_point["point"][:2]
            b_prob = b_point["probability"]
            b_spacing = b_point["point"][2]

            half_conf_b = {
                "name": f"NoMatchB {len(fused_points)}",
                "point": [b_xy[0], b_xy[1], b_spacing],
                "probability": b_prob * 0.25
            }
            fused_points.append(half_conf_b)

    # --- 4) Build final JSON
    fused_data = {
        "name": "fused-points",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": fused_points
    }

    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(fused_data, f, indent=4)

    print(f"Fused JSON saved to {output_json_path}")



import torch
import numpy as np
from PIL import Image
from inference_utils.processing_utils import get_instances
import SimpleITK as sitk

def softmax_scalar(avg_conf: float) -> float:
    """
    Interprets 'avg_conf' as one logit vs. 0.0 as the second logit,
    applies torch.softmax across both, and returns the probability
    for 'avg_conf'.
    """
    # Create a tensor [avg_conf, 0.0] to have at least two values
    logits = torch.tensor([avg_conf, 0.0], dtype=torch.float32)
    
    # Apply softmax along dim=0 for a 1D tensor
    probs = torch.softmax(logits, dim=0)
    
    # probs[0] is the softmax result for 'avg_conf'
    return float(probs[0])

def append_to_json_file(location, new_content):
    """
    Appends new content to an existing JSON file or creates a new one if it doesn't exist.
    """
    # Load existing data if the file exists
    if location.exists():
        with open(location, 'r') as f:
            existing_content = json.load(f)
    else:
        # Initialize an empty structure if file doesn't exist
        existing_content = {
            "name": new_content["name"],
            "type": new_content["type"],
            "version": new_content["version"],
            "points": []
        }

    # Append new points
    existing_content["points"].extend(new_content["points"])

    # Write updated content back to the file
    with open(location, 'w') as f:
        json.dump(existing_content, f, indent=4)
    print(f"Appended predictions to {location}")

# Calculate average confidence for each blob
def calculate_blob_confidences(mask, instance_mask):
    confidences = []
    ins_ids = np.unique(instance_mask)
    for ins_id in ins_ids:
        if ins_id == 0:
            continue
        # Extract pixels belonging to the blob
        blob_pixels = mask[instance_mask == ins_id]
        # Calculate average confidence
        confidences.append(blob_pixels.mean())
    return confidences

def inference_with_biomed_model(iterator, model, spacing, image_path, output_path,
                                json_filename_lymphocytes, json_filename_monocytes, json_filename_inflammatory_cells):
    """
    Feeds patches from 'iterator' into the Biomed model, extracts bounding boxes
    from segmentation masks, and saves the output as JSON. The 'probability'
    field is computed from the average pixel value of each blob in the
    segmentation mask (i.e., average confidence).

    Steps:
      1. For each patch, run Biomed model -> returns up to 3 segmentation masks
         (lymphocytes, monocytes, inflammatory cells).
      2. Threshold & get instance-labeled masks for each category (e.g., 'ins_id').
      3. For each instance:
         - Compute bounding box.
         - Extract pixels (from the raw probability mask) belonging to that instance.
         - Compute average pixel value = final probability.
      4. Save each bounding box as a "point" with 'probability' in JSON.
    """
    print(f"Starting inference for {image_path}...")

    # Prepare output dictionaries
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

    # Spacing ratio
    spacing_min = 0.25
    ratio = spacing / spacing_min

    # Obtain real spacing from WSI
    with WholeSlideImage(image_path) as wsi:
        real_spacing = wsi.get_real_spacing(spacing_min)
        print(f"Real spacing for {image_path}: {real_spacing}")

    patch_count = 0

    # This function returns bounding boxes AND the instance ID in each box
    def instance_masks_to_boxes(instance_mask):
        """
        For each connected component ID in 'instance_mask', return:
         (xmin, ymin, xmax, ymax, instance_id)
        """
        boxes_and_ids = []
        ins_ids = np.unique(instance_mask)
        for ins_id in ins_ids:
            if ins_id == 0:
                continue
            coords = np.column_stack(np.where(instance_mask == ins_id))
            ymin, xmin = coords.min(axis=0)
            ymax, xmax = coords.max(axis=0)
            boxes_and_ids.append((xmin, ymin, xmax, ymax, ins_id))
        return boxes_and_ids

    for x_batch, _, info in iterator:
        x_batch = x_batch.squeeze(0)  # shape: (BatchSize, Height, Width, Channels)
        x_batch = x_batch
        c = info["x"]
        r = info["y"]
        patch_count += 1

        # Convert patch to a PIL Image and predict segmentation masks
        # 'pred_mask' is a list of up to 3 arrays [lymph, mono, inflammatory].
        img = Image.fromarray(x_batch[0, :, :, :][:, :, [2, 1, 0]])  # single patch
        pred_mask = interactive_infer_image(model, img, ['lymphocytes', 'monocytes', 'inflammatory cells'])

        # For each category, threshold & get instance-labeled masks
        instance_masks = []
        for pm in pred_mask:
            # Example threshold = 0.25
            bin_mask = (pm > BPARSET).astype(np.uint8)
            labeled_mask = get_instances(bin_mask)  # each blob has a unique ID
            instance_masks.append((pm, labeled_mask))

        # Categories in same order
        categories = ['lymphocytes', 'monocytes', 'inflammatory cells']

        for cat_idx, (probability_mask, labeled_mask) in enumerate(instance_masks):
            # For each instance in 'labeled_mask'
            boxes_and_ids = instance_masks_to_boxes(labeled_mask)

            for (xmin, ymin, xmax, ymax, ins_id) in boxes_and_ids:
                # Convert bounding box to center coords
                x_center = ((xmin + xmax) / 2.0) * ratio + c
                y_center = ((ymin + ymax) / 2.0) * ratio + r

                # Compute average probability from 'probability_mask'
                blob_pixels = probability_mask[labeled_mask == ins_id]
                avg_conf = float(blob_pixels.mean()) if blob_pixels.size > 0 else 0.0
                #avg_conf = softmax_scalar(avg_conf)
                #print(avg_conf)
                
                # Create the JSON record
                point_record = {
                    "name": f"Point {len(output_dict_lymphocytes['points'])}",
                    "point": [px_to_mm(x_center, real_spacing),
                              px_to_mm(y_center, real_spacing),
                              real_spacing],
                    "probability": avg_conf  # actual average from the segmentation mask
                }

                # Distribute into correct category dictionary
                if cat_idx == 0:  # lymphocytes
                    output_dict_lymphocytes["points"].append(point_record)
                elif cat_idx == 1:  # monocytes
                    output_dict_monocytes["points"].append(point_record)
                else:               # inflammatory cells
                    output_dict_inflammatory_cells["points"].append(point_record)

    print(f"Processed {patch_count} patches for {image_path}.\n"
          f"Lymphocytes: {len(output_dict_lymphocytes['points'])}, "
          f"Monocytes: {len(output_dict_monocytes['points'])}, "
          f"Inflammatory cells: {len(output_dict_inflammatory_cells['points'])}")

    # Save JSON outputs
    print("Saving JSON results...")
    write_json_file(json_filename_lymphocytes, output_dict_lymphocytes)
    write_json_file(json_filename_monocytes, output_dict_monocytes)
    write_json_file(json_filename_inflammatory_cells, output_dict_inflammatory_cells)
    
    # Also save to top-level (if desired)
    top_level_lymph = EVALUATION_PATH / "detected-lymphocytes.json"
    top_level_mono = EVALUATION_PATH / "detected-monocytes.json"
    top_level_inflam = EVALUATION_PATH / "detected-inflammatory-cells.json"

    write_json_file(top_level_lymph, output_dict_lymphocytes)
    write_json_file(top_level_mono, output_dict_monocytes)
    write_json_file(top_level_inflam, output_dict_inflammatory_cells)

    print("JSON files saved.\n")


# -------------------------------------------------------------------------
# Inference Function
# -------------------------------------------------------------------------
def inference(iterator, predictors, spacing, image_path, output_path,
              json_filename_lymphocytes, json_filename_monocytes, json_filename_inflammatory_cells):
    """
    Feeds patches from 'iterator' into each model in 'predictors',
    fuses the predictions, and saves the output as JSON.
    """
    print(f"Starting inference for {image_path}...")

    # Prepare output dicts
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

    # Define normalization
    target_mean = [183.68832 / 255.0, 162.18071 / 255.0, 186.88966 / 255.0]
    target_std = [57.35563 / 255.0, 68.62531 / 255.0, 54.158657 / 255.0]
    normalize_transform = transforms.Normalize(mean=target_mean, std=target_std)

    # Spacing ratio
    spacing_min = 0.25
    ratio = spacing / spacing_min

    # Obtain real spacing from WSI
    with WholeSlideImage(image_path) as wsi:
        real_spacing = wsi.get_real_spacing(spacing_min)
        print(f"Real spacing for {image_path}: {real_spacing}")

    # Patch count
    patch_count = 0

    for x_batch, y_batch, info in iterator:
        x_batch = x_batch.squeeze(0)  # shape: (BatchSize, Height, Width, Channels)
        y_batch = y_batch.squeeze(0)

        c = info["x"]
        r = info["y"]
        patch_count += 1

        # Optional: apply your normalization transform
        # normalized_img_tensor = normalize_transform(
        #    torch.from_numpy(x_batch).permute(0, 3, 1, 2).float() / 255.0
        # ).squeeze(0).tranpose(2, 0, 1)

        # For this pipeline, we can just convert to float in [0,1]
        image_tensor = torch.from_numpy(x_batch).permute(0, 3, 1, 2).float() / 255.0

        # Collect predictions from each model
        boxes_list = []
        scores_list = []
        labels_list = []

        for model in predictors:
            # You can adapt these conf/iou/augment settings per model
            results = model.predict(
                image_tensor,
                imgsz=256,
                max_det=1000,
                conf=YOLOT, #seems best
                iou=0.75,
                augment=True
            )

            # Parse each result (one per batch element if batch>1)
            b_temp, s_temp, l_temp = [], [], []
            for res in results:
                for box in res.boxes:
                    coords = box.data.cpu().numpy()[:, :4][0] / 256.0
                    score = float(box.conf.cpu().numpy()[0])
                    #score = softmax_scalar(score)
                    label = int(box.cls.cpu().numpy()[0])
                    b_temp.append(coords)
                    s_temp.append(score)
                    l_temp.append(label)

            boxes_list.append(b_temp)
            scores_list.append(s_temp)
            labels_list.append(l_temp)

        # Weighted Boxes Fusion
        iou_thr = 0.5
        skip_box_thr = 0.0001
        weights = [1] * len(predictors)

        boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr
        )

        # Convert fused boxes back to 256 scale
        for box, cls, prob in zip(boxes_fused, labels_fused, scores_fused):
            box = box * 256.0
            xmin, ymin, xmax, ymax = box

            # Convert to center
            x_center = ((xmin + xmax) / 2.0) * ratio + c
            y_center = ((ymin + ymax) / 2.0) * ratio + r

            point_record = {
                "name": f"Point {len(output_dict_lymphocytes['points'])}",
                "point": [px_to_mm(x_center, real_spacing), px_to_mm(y_center, real_spacing), real_spacing],
                "probability": prob
            }

            if cls == 0:
                output_dict_lymphocytes["points"].append(point_record)
            elif cls == 1:
                output_dict_monocytes["points"].append(point_record)

            # All inflammatory cells
            output_dict_inflammatory_cells["points"].append(point_record)

    print(f"Processed {patch_count} patches for {image_path}.\n"
          f"Lymphocytes: {len(output_dict_lymphocytes['points'])}, "
          f"Monocytes: {len(output_dict_monocytes['points'])}, "
          f"Inflammatory cells: {len(output_dict_inflammatory_cells['points'])}")

    # Save JSON outputs to both case_output_path and top-level EVALUATION_PATH
    print("Saving JSON results...")
    append_to_json_file(json_filename_lymphocytes, output_dict_lymphocytes)
    append_to_json_file(json_filename_monocytes, output_dict_monocytes)
    append_to_json_file(json_filename_inflammatory_cells, output_dict_inflammatory_cells)

    # Also save to top-level (if desired)
    top_level_lymph = EVALUATION_PATH / "detected-lymphocytes.json"
    top_level_mono = EVALUATION_PATH / "detected-monocytes.json"
    top_level_inflam = EVALUATION_PATH / "detected-inflammatory-cells.json"

    append_to_json_file(top_level_lymph, output_dict_lymphocytes)
    append_to_json_file(top_level_mono, output_dict_monocytes)
    append_to_json_file(top_level_inflam, output_dict_inflammatory_cells)
    print("JSON files saved.\n")


# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------
def write_json_file(location, content):
    """
    Save 'content' dict as a JSON file to 'location'.
    """
    location.parent.mkdir(parents=True, exist_ok=True)
    with open(location, 'w') as f:
        json.dump(content, f, indent=4)


def px_to_mm(px, spacing):
    """
    Convert pixel distance to millimeters based on the 'spacing' (microns/px).
    spacing is in microns per pixel. So 1 px = spacing microns = spacing/1000 mm.
    """
    return px * spacing / 1000.0


# -------------------------------------------------------------------------
# Script Entrypoint
# -------------------------------------------------------------------------
if __name__ == "__main__":
    run()
