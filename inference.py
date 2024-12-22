import os
import json
from glob import glob
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
from tqdm import tqdm
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.iterators import create_patch_iterator, PatchConfiguration
from torchvision import transforms
from ensemble_boxes import weighted_boxes_fusion


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

    # Iterate over each (image, mask) pair
    for image_path, mask_path in zip(image_paths, mask_paths):
        base_name = Path(image_path).stem

        # Output folder for this case
        case_output_path = EVALUATION_PATH / "input" / base_name / "output"
        case_output_path.mkdir(parents=True, exist_ok=True)

        # Define JSON output file names
        json_filename_lymphocytes = case_output_path / "detected-lymphocytes.json"
        json_filename_monocytes = case_output_path / "detected-monocytes.json"
        json_filename_inflammatory_cells = case_output_path / "detected-inflammatory-cells.json"

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
            cpus=4,
            backend='asap'
        )

        # Confirm the iterator is ready
        try:
            _ = next(iterator)
            print(f"Iterator for {base_name} loaded successfully.")
        except StopIteration:
            print(f"Iterator for {base_name} is empty.")
        except Exception as e:
            print(f"Error initializing iterator for {base_name}: {e}")

        # Run inference and save predictions
        inference(
            iterator=iterator,
            predictors=predictors,
            spacing=patch_configuration.spacings[0],
            image_path=image_path,
            output_path=case_output_path,
            json_filename_lymphocytes=json_filename_lymphocytes,
            json_filename_monocytes=json_filename_monocytes,
            json_filename_inflammatory_cells=json_filename_inflammatory_cells
        )

        # Stop the iterator once done
        iterator.stop()

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

    # Save the combined metadata JSON
    metadata_file = EVALUATION_PATH / "input" / "predictions.json"
    write_json_file(location=metadata_file, content=output_metadata)
    print(f"Metadata JSON saved to {metadata_file}")


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
        # )

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
                conf=0.25, #seems best
                iou=0.4,
                augment=True
            )

            # Parse each result (one per batch element if batch>1)
            b_temp, s_temp, l_temp = [], [], []
            for res in results:
                for box in res.boxes:
                    coords = box.data.cpu().numpy()[:, :4][0] / 256.0
                    score = float(box.conf.cpu().numpy()[0])
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
