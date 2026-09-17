import json
from pathlib import Path
import os
import time
from PIL import Image
import numpy as np
import torch

from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor


def ae_mivolo(mivolo_model, image_processor, img, patch):
    age = None
    crop = img.crop(patch)
    crop = np.array(crop)


    # face crops
    faces_crops = [None]  # may be [None] if bodies_crops is not None

    # body crops
    bodies_crops = [crop]  # may be [None] if faces_crops is not None

    # prepare BGR inputs
    faces_input = image_processor(images=faces_crops)["pixel_values"]
    body_input = image_processor(images=bodies_crops)["pixel_values"]

    device = mivolo_model.device
    dtype = mivolo_model.dtype

    faces_input = faces_input.to(dtype=dtype, device=device)
    body_input = body_input.to(dtype=dtype, device=device)

    output = mivolo_model(faces_input=faces_input, body_input=body_input)

    if(len(output.age_output)>0):
        age = int(output.age_output[0].item())

    return age


def estimate_age_from_patches(
                ae_method, filename, imgfilename,
        correct_detection_jsons, all_patches, estimated_patches, adults_patches, minors_patches, suffix):

    fixed_data = []

    try:
        with Image.open(imgfilename) as img:
            width, height = img.size

            print(f"Processing {filename}")

            with open(filename, encoding="utf-8") as f:
                detdata = json.loads(f.read())
                if(detdata is not None):
                    correct_detection_jsons +=1
                    for detentry in detdata:
                        if isinstance(detentry, dict):
                            if 'class' in detentry and detentry['class'] == 'person':
                                all_patches += 1
                                fixedres = detentry
                                age = ae_method(img, detentry['bbox'])
                                if(age is not None):
                                    minor = 'minor' if age <=14 else 'adult'
                                    if(minor=='minor'):
                                        minors_patches += 1
                                    else:
                                        adults_patches += 1
                                    fixedres['label'] = minor
                                    fixedres['age'] = age
                                    estimated_patches += 1
                                else:
                                    fixedres['age'] = -1
                                    fixedres['label'] = 'unknown'

                                fixed_data.append(fixedres)


                    with open(filename.replace( suffix, '_with_age_mivolo.json'), 'w') as outf:
                        json.dump(fixed_data, outf, indent=2)

    except Exception as e:
        print("Error processing file {}: {}".format(filename,e))
    return correct_detection_jsons, all_patches, estimated_patches, adults_patches, minors_patches



def run_age_estimation(imagesdir, outputdir, detection_engine_name, suffix= "_all_detections.json"):

    # dtype = torch.cuda.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    mivolo_model = AutoModelForImageClassification.from_pretrained(
        "iitolstykh/mivolo_v2", trust_remote_code=True, torch_dtype=torch.float16
    )
    image_processor = AutoImageProcessor.from_pretrained(
        "iitolstykh/mivolo_v2", trust_remote_code=True
    )

    mivolo_model.to(device)

    ae_method = lambda image, patch: ae_mivolo(mivolo_model, image_processor, image, patch)

    root_outputdir = Path(outputdir, detection_engine_name)

    correct_detection_jsons = 0
    all_patches = 0
    estimated_patches = 0
    adults_patches = 0
    minors_patches = 0
    estimate_age_total_s = 0.0
    estimate_age_calls = 0
    for filename in root_outputdir.rglob('*' + suffix):
        if filename.is_file():
            relpath = os.path.split(os.path.relpath(filename, root_outputdir))[-2]
            stem = os.path.basename(str(filename))[: -len(suffix)]
            imgfilename = os.path.join(imagesdir, relpath, stem + ".jpg")
            t0 = time.perf_counter()
            correct_detection_jsons, all_patches, estimated_patches, adults_patches, minors_patches = estimate_age_from_patches(
                ae_method, str(filename), imgfilename, correct_detection_jsons, all_patches, estimated_patches,
                adults_patches, minors_patches, suffix)
            estimate_age_total_s += (time.perf_counter() - t0)
            estimate_age_calls += 1

    avg_s = (estimate_age_total_s / estimate_age_calls) if estimate_age_calls else 0.0
    summary = (
        f"Correct entries JSONS in {correct_detection_jsons} results. "
        f"For {estimated_patches}/{all_patches} bboxes age is estimated. "
        f"{adults_patches} are adults, {minors_patches} are minors. "
        f"{all_patches - estimated_patches} are uknown. "
        f"Avg processing time: {avg_s:.4f}s over {estimate_age_calls} files."
    )
    print(summary)
    summary_path = os.path.join(root_outputdir, "summary_ae_mivolo.txt")
    with open(summary_path, "w", encoding="utf-8") as outf:
        outf.write(summary + "\n")

if __name__ == '__main__':



    outputdir = os.environ.get('OUTPUTDIR', '')
    imagesdir = os.environ.get('IMAGESDIR', '')

    detection_engine_name = 'detector_yolo26x'


    run_age_estimation(imagesdir, outputdir, detection_engine_name)






