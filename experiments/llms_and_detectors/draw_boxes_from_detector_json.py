import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os

from pycocotools.coco import COCO
from evaluate_coco_detection import get_bboxes_for_image




def _try_load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_detections(im: Image.Image, data: dict, color='green') -> None:
    draw = ImageDraw.Draw(im)
    width, height = im.size
    font = _try_load_font(14)
    green = (0, 220, 0)
    black = (0, 0, 0)

    if('detections' in data):
        detections = data['detections']
    elif(len(data)>0):
        detections = data
    else:
        detections = []
    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)

        if 'label' in det:
            label = str(det.get("label", ""))
            age = int(det.get("age", -1))
            text = f"{label}"# {age}"
        elif 'class' in det:
            label = str(det.get("class", ""))
            conf = float(det.get("confidence", 0.0))
            text = f"{label}"# {conf:.2f}"
        elif 'description' in det:
            text = det.get("description", "")[:16]
            lastspace = text.rfind(" ")
            text = text[:lastspace] if lastspace != -1 else text
        else:
            text = ""

        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)

        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top
        ty = max(y1 - 4, th + 4)
        pad = 2
        draw.rectangle(
            [x1, ty - th - 2 * pad, x1 + tw + 2 * pad, ty + pad],
            fill=green,
        )
        draw.text((x1 + pad, ty - th - pad), text, fill=black, font=font)

def process_files(imagesdir, outputdir, det_engine_name, gt_path=None, suffix = "_fixed.json"):

    coco_gt = None
    if (gt_path is not None):
        coco_gt = COCO(str(gt_path))


    root_outputdir = Path(outputdir, det_engine_name)

    processed_images = 0
    missing_images = 0
    bad_jsons = 0
    for filename in root_outputdir.rglob('*' + suffix):
        if filename.is_file():
            relpath = os.path.split(os.path.relpath(filename, root_outputdir))[-2]
            stem = os.path.basename(str(filename))[: -len(suffix)]

            try:
                with open(filename, "r", encoding="utf-8") as f:
                    data = json.loads(f.read())
            except json.JSONDecodeError as e:
                print(f"{filename}: invalid JSON ({e})", file=sys.stderr)
                bad_jsons += 1
                continue
            srcframe = Path(imagesdir, relpath, stem + ".jpg")
            if srcframe is None:
                print(
                    f"{filename}: no JPEG found for stem {stem!r} ",
                    file=sys.stderr,
                )
                missing_images += 1
                continue

            try:
                img = Image.open(srcframe).convert("RGB")

                draw_detections(img, data)
                if (coco_gt is not None):
                    gt_anns = get_bboxes_for_image(coco_gt, str(srcframe))
                    for ann in gt_anns:
                        ann['bbox'] = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2],
                                       ann['bbox'][1] + ann['bbox'][3]]
                    draw_detections(img, gt_anns, color='red')

                print(f"Writing image for file {filename}")
                img.save(str(filename).replace(suffix, ".jpg"))
                processed_images += 1
            except OSError as e:
                print(f"{srcframe}: cannot open ({e})", file=sys.stderr)
                missing_images += 1
                continue

    print(
        f"Done: {processed_images} written, {missing_images} missing image/read fail, {bad_jsons} bad JSON"
    )


if __name__ == "__main__":
    outputdir = os.environ.get('OUTPUTDIR', '')
    imagesdir = os.environ.get('IMAGESDIR', '')

    gt_path = None#("../../data/parenting_slop_dataset/annotations/parenting_slop_dataset.json")


    # engine = 'llm_gemma-3-4b-it'
    # engine = 'llm_gemma-4-E4B-it'
    # engine = 'llm_Qwen3.5-4B'

    # engine = 'llm_Qwen3.6-35B-A3B'
    # engine = 'llm_Qwen3.6-35B-A3B_chonly'

    # engine = 'detector_yoloe-26x-seg'
    # engine = 'detector_yolo26x'
    # engine = 'detector_yolov3u'
    engine = 'detector_yolov8x-world'


    SUFFIX = "_fixed.json"

    process_files(imagesdir,outputdir,engine, gt_path, SUFFIX)