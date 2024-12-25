import os
import json
import sys
from glob import glob
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.iterators import create_patch_iterator, PatchConfiguration

# -------------------------------------------------------------------------
# Add required Python paths
# -------------------------------------------------------------------------
current_script_path = os.path.dirname(os.path.abspath(__file__))
if current_script_path not in sys.path:
    sys.path.append(current_script_path)

sys.path.append("/opt/app/biomp")
sys.path.append("/opt/app/biomp/modeling")
sys.path.append("/opt/app/biomp/utilities")
sys.path.append("/opt/app/biomp/inference_utils")
sys.path.append("/opt/app/")

# -------------------------------------------------------------------------
# Environment Variables for Offline Mode
# -------------------------------------------------------------------------
os.environ["HF_HOME"] = "/opt/app/cache/huggingface"
os.environ["HUGGINGFACE_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# -------------------------------------------------------------------------
# Biomp Imports
# -------------------------------------------------------------------------
import biomp
import biomp.utilities
from biomp.modeling.BaseModel import BaseModel
from biomp.modeling import build_model
from biomp.utilities.distributed import init_distributed
from biomp.utilities.arguments import load_opt_from_config_files
from biomp.utilities.constants import BIOMED_CLASSES
from biomp.inference_utils.inference import interactive_infer_image
from inference_utils.processing_utils import get_instances

# -------------------------------------------------------------------------
# Global Paths
# -------------------------------------------------------------------------
INPUT_PATH = Path("/input")
EVALUATION_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")

# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------
def write_json_file(location: Path, content: dict) -> None:
    """
    Saves `content` as a JSON file to `location`.
    """
    location.parent.mkdir(parents=True, exist_ok=True)
    with open(location, "w") as f:
        json.dump(content, f, indent=4)

def read_json_file(location: Path) -> dict:
    """
    Reads JSON content from a file. Returns an empty structure if not found.
    """
    if not location.exists():
        return {}
    with open(location, "r") as f:
        return json.load(f)

def px_to_mm(px: float, spacing: float) -> float:
    """
    Convert pixel distance to millimeters based on the 'spacing' in microns/px.
    """
    return px * spacing / 1000.0

def distance_2d(p1: list, p2: list) -> float:
    """Euclidean distance between two [x, y] points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def fuse_points_by_radius(points_a, points_b, radius=0.05):
    """
    Fuse two lists of points based on proximity (<= radius) and confidence.
    """
    fused = []
    b_used = set()
    
    for a_pt in points_a:
        a_xy = a_pt["point"][:2]  # (x_mm, y_mm)
        a_prob = a_pt["probability"]
        best_match = None
        best_match_idx = None
        best_dist = float("inf")

        for idx_b, b_pt in enumerate(points_b):
            if idx_b in b_used:
                continue
            b_xy = b_pt["point"][:2]
            dist = distance_2d(a_xy, b_xy)
            if dist < radius and dist < best_dist:
                best_dist = dist
                best_match = b_pt
                best_match_idx = idx_b

        if best_match is not None:
            # We have a match within radius: choose higher probability
            if a_prob >= best_match["probability"]:
                fused.append(a_pt)
            else:
                fused.append(best_match)
            b_used.add(best_match_idx)
        else:
            # No match found, keep the point from A
            fused.append(a_pt)

    # Add leftover points from B
    for idx_b, b_pt in enumerate(points_b):
        if idx_b not in b_used:
            fused.append(b_pt)

    return fused

# -------------------------------------------------------------------------
# Inference: Save 1024 and 256 Predictions to EVALUATION_PATH
# -------------------------------------------------------------------------
def inference_with_biomed_model(
    iterator,
    model,
    spacing: float,
    image_path: str,
    suffix="_1024",
):
    """
    Runs Biomed (1024) inference, saves to EVALUATION_PATH/detected-<category>_<suffix>.json
    """
    print(f"Starting Biomed inference for {image_path} with suffix {suffix}...")

    output_dict_lymph = {
        "name": f"lymphocytes{suffix}",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }
    output_dict_mono = {
        "name": f"monocytes{suffix}",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }
    output_dict_inflam = {
        "name": f"inflammatory-cells{suffix}",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    spacing_min = 0.25
    ratio = spacing / spacing_min

    with WholeSlideImage(image_path) as wsi:
        real_spacing = wsi.get_real_spacing(spacing_min)
        print(f"Real spacing: {real_spacing}")

    patch_count = 0

    def instance_masks_to_boxes(mask: np.ndarray):
        boxes = []
        ins_ids = np.unique(mask)
        for ins_id in ins_ids:
            if ins_id == 0:
                continue
            coords = np.column_stack(np.where(mask == ins_id))
            ymin, xmin = coords.min(axis=0)
            ymax, xmax = coords.max(axis=0)
            boxes.append((xmin, ymin, xmax, ymax))
        return boxes

    for x_batch, _, info in iterator:
        x_batch = x_batch.squeeze(0)
        c, r = info["x"], info["y"]
        patch_count += 1

        img = Image.fromarray(x_batch[0])  # Single patch
        pred_mask = interactive_infer_image(model, img, ["lymphocytes", "monocytes", "inflammatory cells"])
        instance_masks = [get_instances((m > 0.2).astype(np.uint8)) for m in pred_mask]

        for cat_idx, mask_data in enumerate(instance_masks):
            boxes = instance_masks_to_boxes(mask_data)
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                x_center = ((xmin + xmax) / 2.0) * ratio + c
                y_center = ((ymin + ymax) / 2.0) * ratio + r

                record = {
                    "name": f"Point {len(output_dict_lymph['points'])}",
                    "point": [
                        px_to_mm(x_center, real_spacing),
                        px_to_mm(y_center, real_spacing),
                        real_spacing
                    ],
                    "probability": 1.0
                }

                if cat_idx == 0:
                    output_dict_lymph["points"].append(record)
                elif cat_idx == 1:
                    output_dict_mono["points"].append(record)
                else:
                    output_dict_inflam["points"].append(record)

    print(f"Processed {patch_count} patches for {image_path} (Biomed {suffix}).")

    # Write separate JSON directly to EVALUATION_PATH
    write_json_file(EVALUATION_PATH / f"detected-lymphocytes{suffix}.json", output_dict_lymph)
    write_json_file(EVALUATION_PATH / f"detected-monocytes{suffix}.json", output_dict_mono)
    write_json_file(EVALUATION_PATH / f"detected-inflammatory-cells{suffix}.json", output_dict_inflam)

def inference_yolo_256(
    iterator,
    predictors: list,
    spacing: float,
    image_path: str,
    suffix="_256",
):
    """
    Runs YOLO (256) inference + WBF, saves to EVALUATION_PATH/detected-<category>_<suffix>.json
    """
    print(f"Starting YOLO inference for {image_path} with suffix {suffix}...")

    output_dict_lymph = {
        "name": f"lymphocytes{suffix}",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }
    output_dict_mono = {
        "name": f"monocytes{suffix}",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }
    output_dict_inflam = {
        "name": f"inflammatory-cells{suffix}",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    spacing_min = 0.25
    ratio = spacing / spacing_min

    with WholeSlideImage(image_path) as wsi:
        real_spacing = wsi.get_real_spacing(spacing_min)
        print(f"Real spacing: {real_spacing}")

    patch_count = 0

    for x_batch, _, info in iterator:
        x_batch = x_batch.squeeze(0)
        c, r = info["x"], info["y"]
        patch_count += 1

        image_tensor = torch.from_numpy(x_batch).permute(0, 3, 1, 2).float() / 255.0

        boxes_list, scores_list, labels_list = [], [], []

        for mdl in predictors:
            results = mdl.predict(
                image_tensor,
                imgsz=256,
                max_det=1000,
                conf=0.35,
                iou=0.75,
                augment=True
            )
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
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=[1]*len(predictors),
            iou_thr=0.5,
            skip_box_thr=0.0001
        )
        fused_boxes *= 256.0

        for box, cls, prob in zip(fused_boxes, fused_labels, fused_scores):
            xmin, ymin, xmax, ymax = box
            x_center = ((xmin + xmax) / 2.0) * ratio + c
            y_center = ((ymin + ymax) / 2.0) * ratio + r

            record = {
                "name": f"Point {len(output_dict_lymph['points'])}",
                "point": [
                    px_to_mm(x_center, real_spacing),
                    px_to_mm(y_center, real_spacing),
                    real_spacing
                ],
                "probability": prob
            }

            if cls == 0:
                output_dict_lymph["points"].append(record)
            elif cls == 1:
                output_dict_mono["points"].append(record)

            output_dict_inflam["points"].append(record)

    print(f"Processed {patch_count} patches for {image_path} (YOLO {suffix}).")

    # Write separate JSON directly to EVALUATION_PATH
    write_json_file(EVALUATION_PATH / f"detected-lymphocytes{suffix}.json", output_dict_lymph)
    write_json_file(EVALUATION_PATH / f"detected-monocytes{suffix}.json", output_dict_mono)
    write_json_file(EVALUATION_PATH / f"detected-inflammatory-cells{suffix}.json", output_dict_inflam)

# -------------------------------------------------------------------------
# Final Step: Fuse 1024 + 256 into EVALUATION_PATH/detected-*.json
# -------------------------------------------------------------------------
def fuse_final_points(
    category_name: str,
    suffix_1024: str,
    suffix_256: str,
    radius=0.15
):
    """
    Reads two JSON files from EVALUATION_PATH (e.g., detected-lymphocytes_1024.json
    and detected-lymphocytes_256.json), fuses them, and writes the result as
    EVALUATION_PATH/detected-lymphocytes.json (no suffix).
    """
    json_1024 = EVALUATION_PATH / f"detected-{category_name}{suffix_1024}.json"
    json_256 = EVALUATION_PATH / f"detected-{category_name}{suffix_256}.json"
    output_fused = EVALUATION_PATH / f"detected-{category_name}.json"

    data_1024 = read_json_file(json_1024)
    data_256 = read_json_file(json_256)

    if "points" not in data_1024 or "points" not in data_256:
        print(f"Cannot fuse. Missing 'points' in {json_1024} or {json_256}.")
        return

    fused_points = fuse_points_by_radius(data_1024["points"], data_256["points"], radius=radius)

    # Build final structure with same metadata but no suffix in 'name'
    fused_data = {
        "name": category_name,
        "type": data_1024["type"],
        "version": data_1024["version"],
        "points": fused_points
    }
    write_json_file(output_fused, fused_data)
    print(f"Fused {data_1024['name']} with {data_256['name']} -> {output_fused}")

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def run() -> None:
    """
    1. Locate .tif files in input
    2. Load Biomed+YOLO
    3. 1024 & 256 inference -> EVALUATION_PATH/detected-lymphocytes_1024.json, etc.
    4. Fuse pairs -> EVALUATION_PATH/detected-lymphocytes.json, etc.
    5. Save final metadata
    """
    image_paths = sorted(glob(str(INPUT_PATH / "images/kidney-transplant-biopsy-wsi-pas/*.tif")))
    mask_paths = sorted(glob(str(INPUT_PATH / "images/tissue-mask/*.tif")))

    output_metadata = []

    # YOLO models
    weights = ["bestPAS256.pt", "best_PASIHC256.pt", "best2k_PAS256.pt"]
    print("Loading YOLO models...")
    predictors = []
    for w in weights:
        mdl = YOLO(str(MODEL_PATH / w))
        mdl = torch.compile(mdl)
        predictors.append(mdl)
    print(f"Loaded {len(predictors)} YOLO models.")

    # Biomed model
    print("Loading Biomed model...")
    opt = load_opt_from_config_files(["/opt/app/biomp/configs/biomedparse_inference.yaml"])
    init_distributed(opt)
    pretrained_pth = MODEL_PATH / "model_state_dict.pt"
    biomed_model = BaseModel(opt, build_model(opt)).from_pretrained(str(pretrained_pth)).eval().cuda()
    biomed_model = torch.compile(biomed_model)
    with torch.no_grad():
        biomed_model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)
    print("Biomed model loaded.\n")

    # Process each WSI
    for image_path, mask_path in zip(image_paths, mask_paths):
        base_name = Path(image_path).stem

        # 1) Biomed (1024)
        config_1024 = PatchConfiguration(
            patch_shape=(1024, 1024, 3),
            spacings=(0.5,),
            overlap=(0, 0),
            offset=(0, 0),
            center=False
        )
        iterator_1024 = create_patch_iterator(image_path, mask_path, config_1024, cpus=1, backend="asap")
        try:
            next(iterator_1024)
        except StopIteration:
            print(f"Iterator (1024) empty for {base_name}")
        except Exception as e:
            print(f"Error (1024) for {base_name}: {e}")

        inference_with_biomed_model(iterator_1024, biomed_model, spacing=0.5, image_path=image_path)
        iterator_1024.stop()
        del iterator_1024

        # 2) YOLO (256)
        config_256 = PatchConfiguration(
            patch_shape=(256, 256, 3),
            spacings=(0.5,),
            overlap=(0, 0),
            offset=(0, 0),
            center=False
        )
        iterator_256 = create_patch_iterator(image_path, mask_path, config_256, cpus=1, backend="asap")
        try:
            next(iterator_256)
        except StopIteration:
            print(f"Iterator (256) empty for {base_name}")
        except Exception as e:
            print(f"Error (256) for {base_name}: {e}")

        inference_yolo_256(iterator_256, predictors, spacing=0.5, image_path=image_path)
        iterator_256.stop()
        del iterator_256

        # 3) Final Fuse from EVALUATION_PATH
        fuse_final_points("lymphocytes", "_1024", "_256", radius=0.15)
        fuse_final_points("monocytes", "_1024", "_256", radius=0.15)
        fuse_final_points("inflammatory-cells", "_1024", "_256", radius=0.15)

        # 4) Build final metadata
        output_metadata.append({
            "pk": base_name,
            "inputs": [
                {
                    "image": {"name": Path(image_path).name},
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

    del biomed_model

    # 5) Save combined metadata JSON in EVALUATION_PATH/input/predictions.json
    metadata_file = EVALUATION_PATH / "input" / "predictions.json"
    write_json_file(metadata_file, output_metadata)
    print(f"Metadata JSON saved to {metadata_file}")


if __name__ == "__main__":
    run()
