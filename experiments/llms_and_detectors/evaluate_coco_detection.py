from __future__ import annotations
import json
from pathlib import Path
import os
from prettytable import PrettyTable
from enum import Enum

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as exc:
    raise SystemExit(
        "pycocotools is required. Install with: pip install pycocotools"
    ) from exc




def image_rel_key(file_name: str) -> str:
    """``.../frames/0015/frame000000.jpg`` -> ``0015/frame000000.jpg``."""
    path = Path(file_name)
    return f"{path.parent.name}/{path.name}"


def build_gt_lookups(coco: COCO) -> tuple[dict[str, int], dict[int, tuple[int, int]]]:
    rel_to_image_id: dict[str, int] = {}
    image_sizes: dict[int, tuple[int, int]] = {}
    for image_id, meta in coco.imgs.items():
        rel_to_image_id[image_rel_key(meta["file_name"])] = image_id
        image_sizes[image_id] = (int(meta["width"]), int(meta["height"]))
    return rel_to_image_id, image_sizes


def get_bboxes_for_image(coco: COCO, image_file_name: str) -> list[list[float]]:
    """Return COCO xywh bboxes for all annotations of ``image_file_name``.

    ``image_file_name`` may be an absolute path, relative path, or ``parent/name``
    key; images are matched via ``image_rel_key``. Returns an empty list if the
    image is not found.
    """
    target_key = image_rel_key(image_file_name)
    image_id = next(
        (
            img_id
            for img_id, meta in coco.imgs.items()
            if image_rel_key(meta["file_name"]) == target_key
        ),
        None,
    )
    if image_id is None:
        return []
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
    return anns


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [float(x1), float(y1), float(w), float(h)]

def iter_detection_dicts(data: object) -> list[dict]:
    if isinstance(data, dict) and "detections" in data:
        items = data["detections"]
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict) and data:
        items = [data]
    else:
        return []

    detections: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if any(isinstance(k, str) and k.startswith("bbox") for k in item):
            detections.append(item)
            continue
        if len(item) == 1:
            inner = next(iter(item.values()))
            if isinstance(inner, dict):
                detections.append(inner)
    return detections


def bbox_from_detection(det: dict) -> list[float] | None:
    bbox_key = next((k for k in det if isinstance(k, str) and k.startswith("bbox")), None)
    if bbox_key is None:
        return None
    bbox = det[bbox_key]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    return [float(v) for v in bbox]

def normalize_category(label: str) -> int:
    if label == "child" or label == "minor":
        return 0
    elif(label == "person"): #classic detectors as a baseline
        return 0
    elif(label == 'unknown'):
        return 0
    elif label == "adult":
        return 1
    else:
        raise ValueError(label)



def label_from_detection(det: dict) -> str:
    for key in ("label", "class"):
        if key in det and det[key] is not None:
            return str(det[key])
    return ""


def score_from_detection(det: dict) -> float:
    for key in ("confidence", "score"):
        if key in det:
            return float(det[key])
    return 1.0


def prediction_path_to_image_key(pred_path: Path, suffix: str) -> str:
    stem = pred_path.name[: -len(suffix)] if pred_path.name.endswith(suffix) else pred_path.stem
    return f"{pred_path.parent.name}/{stem}.jpg"


def load_predictions(
    predictions_dir: Path,
    suffix: str,
    rel_to_image_id: dict[str, int],
    image_sizes: dict[int, tuple[int, int]],
    *,
    min_score: float,
) -> tuple[list[dict], dict[str, int]]:
    stats = {
        "files_seen": 0,
        "files_parsed": 0,
        "files_no_gt": 0,
        "files_bad_json": 0,
        "detections_kept": 0,
        "detections_skipped_label": 0,
        "detections_skipped_score": 0,
        "detections_skipped_bbox": 0,
    }
    results: list[dict] = []

    for pred_path in sorted(predictions_dir.rglob(f"*{suffix}")):
        if not pred_path.is_file():
            continue
        stats["files_seen"] += 1
        image_key = prediction_path_to_image_key(pred_path, suffix)
        image_id = rel_to_image_id.get(image_key)
        if image_id is None:
            stats["files_no_gt"] += 1
            continue

        try:
            data = json.loads(pred_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stats["files_bad_json"] += 1
            continue

        stats["files_parsed"] += 1
        width, height = image_sizes[image_id]

        for det in iter_detection_dicts(data):
            label = label_from_detection(det)
            category_id = normalize_category(label)
            # if filter_catids is not None and category_id not in filter_catids:
            #     stats["detections_skipped_label"] += 1
            #     continue

            score = score_from_detection(det)
            if score < min_score:
                stats["detections_skipped_score"] += 1
                continue

            bbox = bbox_from_detection(det)
            if bbox is None:
                stats["detections_skipped_bbox"] += 1
                continue

            x1, y1, x2, y2 = bbox
            coco_bbox = xyxy_to_xywh(x1, y1, x2, y2)
            if coco_bbox[2] <= 0 or coco_bbox[3] <= 0:
                stats["detections_skipped_bbox"] += 1
                continue



            results.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": coco_bbox,
                    "score": score,
                }
            )
            stats["detections_kept"] += 1

    return results, stats


def compile_detection_results(
    coco_gt: COCO,
    predictions_dir: Path | str,
    *,
    suffix: str = "_fixed.json",
    min_score: float = 0.0,
) -> tuple[list[dict], dict[str, int]]:

    predictions_dir = Path(predictions_dir)
    rel_to_image_id, image_sizes = build_gt_lookups(coco_gt)

    return load_predictions(
        predictions_dir,
        suffix,
        rel_to_image_id,
        image_sizes,
        min_score=min_score
    )




def run_coco_eval(coco_gt: COCO, results: list[dict]) -> COCOeval:
    coco_dt = coco_gt.loadRes(results if results else [])
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator


def format_metrics(evaluator: COCOeval) -> dict[str, float]:
    names = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    return {
        name: float(value)
        for name, value in zip(names, evaluator.stats[: len(names)])
    }


def bbox_iou_xywh(box_a: list[float], box_b: list[float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union = aw * ah + bw * bh - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def calculate_precision_recall(
    coco_gt: COCO,
    results: list[dict],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Simple detection precision, recall, and F1 at a fixed IoU threshold.

    Per image, predictions are greedily matched to unmatched ground-truth boxes
    (highest score first). A match requires IoU >= ``iou_threshold`` and the
    same ``category_id``.

    Returns precision (TP / (TP + FP)), recall (TP / (TP + FN)), F1, and counts.
    """
    gt_by_image: dict[int, list[dict]] = {}
    for ann in coco_gt.anns.values():
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    preds_by_image: dict[int, list[dict]] = {}
    for pred in results:
        preds_by_image.setdefault(pred["image_id"], []).append(pred)

    tp = fp = fn = 0
    for image_id, gt_anns in gt_by_image.items():
        preds = sorted(
            preds_by_image.get(image_id, []),
            key=lambda det: det["score"],
            reverse=True,
        )
        matched_gt: set[int] = set()

        for pred in preds:
            best_iou = 0.0
            best_gt_idx: int | None = None
            for idx, ann in enumerate(gt_anns):
                if idx in matched_gt:
                    continue
                if pred["category_id"] != ann["category_id"]:
                    continue
                iou = bbox_iou_xywh(pred["bbox"], ann["bbox"])
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_gt_idx is None:
                fp += 1
            else:
                matched_gt.add(best_gt_idx)
                tp += 1

        fn += len(gt_anns) - len(matched_gt)

    for image_id, preds in preds_by_image.items():
        if image_id not in gt_by_image:
            fp += len(preds)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def eval(outputdir, engine, ae, coco_gt, suffix, conf_th=0):
    predictions_path = Path(outputdir, engine)
    results_path = Path(predictions_path, "detresults" + ('_' + ae if ae != '' else '') + ".json")

    results, stats = compile_detection_results(
        coco_gt,
        predictions_path,
        suffix=suffix,
        min_score=conf_th,
    )

    dump_dict = {"images": coco_gt.imgs, "categories": coco_gt.cats, "annotations": results}
    with open(str(results_path), 'w', encoding="utf-8") as output_path:
        json.dump(dump_dict, output_path, indent=2)

    print(f"GT images: {len(coco_gt.imgs)}")
    print(f"GT annotations: {len(coco_gt.anns)}")
    print(
        "Predictions: "
        f"{stats['detections_kept']} boxes from "
        f"{stats['files_parsed']}/{stats['files_seen']} files "
        f"({stats['files_no_gt']} without GT match, "
        f"{stats['files_bad_json']} bad JSON)"
    )
    if stats["detections_skipped_label"]:
        print(f"Skipped by label filter: {stats['detections_skipped_label']}")
    if stats["detections_skipped_score"]:
        print(f"Skipped by score threshold: {stats['detections_skipped_score']}")
    if stats["detections_skipped_bbox"]:
        print(f"Skipped due to missing/invalid bbox: {stats['detections_skipped_bbox']}")

    print(f"Wrote predictions to {results_path}")

    if not results:
        print("No predictions to evaluate.")

        return [engine, ae, "-", "-", "-", "-", "-", "-", "-"]

    print("\nCOCO bbox metrics for {}:".format(engine))
    evaluator = run_coco_eval(coco_gt, results)
    metrics = format_metrics(evaluator)
    pr_metrics = calculate_precision_recall(coco_gt, results)

    row = [
            engine,
            ae,
            f"{metrics['AP']:.3f}",
            f"{metrics['AP50']:.3f}",
            f"{metrics['AP75']:.3f}",
            # f"{metrics['AR1']:.3f}",
            # f"{metrics['AR10']:.3f}",
            f"{metrics['AR100']:.3f}",
            f"{pr_metrics['precision']:.3f}",
            f"{pr_metrics['recall']:.3f}",
            f"{pr_metrics['f1']:.3f}"
        ]
    return row




if __name__ == "__main__":
    OUTPUTDIR = os.environ.get('OUTPUTDIR', '')

    CONF_TH = 0
    # ground-truth labels are in COCO format, coordinates are in xywh format, they have one category "minors" with id=0
    # detection results for classic detectors are written one-file-per-image, they have category "person" and files with additional age estimation can have label adult|child|unkonwn
    # detection results for llms are written in one file-per-image, they have label child|adult
    gt_path = ("../../data/parenting_slop_dataset/annotations/parenting_slop_dataset.json")
    coco_gt = COCO(str(gt_path)) #normalized, only minors labels


    summary_table = PrettyTable()
    summary_table.field_names = [
        "engine",
        "age-estimation",
        "AP",
        "AP50",
        "AP75",
        # "AR1",
        # "AR10",
        "AR100",
        "global-P",
        "global-R",
        "global-F1"
    ]

    ae = ''

    engines = [
            # "llm_Qwen3.5-4B",
            "llm_Qwen3.6-35B-A3B",
            "llm_Qwen3.6-27B",
            #    "llm_gemma-3-4b-it",
            #    "llm_gemma-4-E4B-it",
            #    'detector_yolov3u',
            #    'detector_yolo11x',
               'detector_yolo26x',
        # 'detector_rfdetr_l',
        # 'detector_retinanet_r50_fpn_2x_coco',
        #         'detector_yoloe-26x-seg',
        # 'detector_yolov8x-world',
        # 'detector_faster-rcnn_r50_fpn_1x_coco',
        # 'detector_cascade-mask-rcnn_r50_fpn_1x_coco',
        # 'detector_mask-rcnn_r50_fpn_1x_coco',

        #        "llm_gemma-4-E4B-it_chonly",
        #     "llm_Qwen3.5-4B_chonly",
        # "llm_Qwen3.6-35B-A3B_chonly"
               ]



    for engine in engines:
        for ae in ['',  'mivolo', 'insightface', 'llm_Qwen3.5-4B', 'llm_Qwen3.6-27B', 'llm_Qwen3.6-35B-A3B', 'llm_p3_Qwen3.6-35B-A3B']:
        # for filter_mode in [CatMode.minors_to_minors, CatMode.ignore_cats_take_all]:

            if engine.startswith("llm_") and ae!='':
                continue
            # if(ae!=''):
            #     filter_mode = CatMode.minors_to_minors_and_unknown

            SUFFIX  = "_with_age"+(("_"+ae) if ae != "" else "")+ ".json" if ("detector" in engine  and ae!="") else "_fixed.json"

            row = eval(OUTPUTDIR, engine, ae, coco_gt,  SUFFIX, CONF_TH)
            summary_table.add_row(row)

    print("\nSummary:")
    print(summary_table.get_formatted_string('csv'))



