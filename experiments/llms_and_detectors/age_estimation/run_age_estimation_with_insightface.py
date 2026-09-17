import json
from pathlib import Path
import os
import time
from PIL import Image
import numpy as np

from insightface.app import FaceAnalysis

def ae_insightface(model, img, patch):
    age = None
    crop = img.crop(patch)
    crop = np.array(crop)

    # crop.show()
    faces = model.get(crop)
    if(len(faces)>0):
        age = faces[0].age
    return age


def estimate_age_from_patches(
                ae_model, filename, imgfilename,
        correct_detection_jsons, all_patches, estimated_patches, adults_patches, minors_patches):

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


                    with open(filename.replace( SUFFIX, '_with_age_insightface.json'), 'w') as outf:
                        json.dump(fixed_data, outf, indent=2)

    except Exception as e:
        print("Error processing file {}: {}".format(filename,e))
    return correct_detection_jsons, all_patches, estimated_patches, adults_patches, minors_patches

def run_age_estimation(imagesdir, outputdir, detection_engine_name, suffix= "_all_detections.json"):
    ifmodel = FaceAnalysis(allowed_modules=['detection', 'genderage'])
    ifmodel.prepare(ctx_id=0, det_size=(640, 640))

    ae_method = lambda image, patch: ae_insightface(ifmodel, image, patch)

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
                adults_patches, minors_patches)
            estimate_age_total_s += (time.perf_counter() - t0)
            estimate_age_calls += 1
        if (all_patches > 100):
            break
    avg_s = (estimate_age_total_s / estimate_age_calls) if estimate_age_calls else 0.0
    summary = (
        f"Correct entries JSONS in {correct_detection_jsons} results. "
        f"For {estimated_patches}/{all_patches} bboxes age is estimated. "
        f"{adults_patches} are adults, {minors_patches} are minors. "
        f"{all_patches - estimated_patches} are uknown. "
        f"Avg processing time: {avg_s:.4f}s over {estimate_age_calls} files."
    )
    print(summary)
    summary_path = os.path.join(root_outputdir, "summary_ae_insightface.txt")
    with open(summary_path, "w", encoding="utf-8") as outf:
        outf.write(summary + "\n")


if __name__ == '__main__':



    outputdir = os.environ.get('OUTPUTDIR', '')
    imagesdir = os.environ.get('IMAGESDIR', '')

    detection_engine_name = 'sandbox_yolo26x'

    run_age_estimation(imagesdir, outputdir, detection_engine_name)





