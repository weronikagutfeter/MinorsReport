import json
from pathlib import Path
import os
import time
from PIL import Image
import numpy as np
from io import BytesIO
from PIL import Image
import base64
from openai import OpenAI


class LLMEstimator(object):
    llm_model_name = None
    prompt_version = 0
    def __init__(self, port, prompt_version=5):

        self.port = port
        self.client = OpenAI(
            base_url="http://127.0.0.1:{}/v1".format(self.port),
            api_key="sk-no-key-required"
        )
        self.prompt_version = prompt_version
        models = self.client.models.list().data
        if(len(models)>0):
            self.llm_model_name = models[0].id.replace('-UD-Q4_K_XL.gguf', '')
        else:
            raise ValueError("No models found")

    def __call__(self, img, bbox, quality=90):

        if(self.prompt_version!=5):
            patch = img.crop(bbox)
        else:
            patch = img
        buf = BytesIO()
        patch.save(buf, format="JPEG", quality=int(quality), optimize=True)

        base64_patch = base64.b64encode(buf.getvalue()).decode("utf-8")

        if (self.prompt_version == 0):
            input = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text",
                         "text": "What age is person in the image? Result should be in either adult (for person older than 14) or minor (for person younger than 14). If there is no person in the image, return unknown."},

                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_patch}",
                        },
                    ],
                }
            ]
        elif(self.prompt_version == 1):
            input = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text",
                         "text": "Do you see any child under 14 years old in the picture? If yes the answer should be 'minor', if you see only older people answer should be 'adult'. Otherwise, if there is no person in the image, return 'unknown'."},

                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_patch}",
                        },
                    ],
                }
            ]
        elif(self.prompt_version == 3):
            input = [
                {"role": "system",
                 "content": "You are an age estimation model. Return age as number for the person visible in the input image"},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text",
                         "text": "What age is person in the image? Result should be in either adult (for person older than 14) or minor (for person younger than 14). If there is no person in the image, return unknown."},

                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_patch}",
                        },
                    ],
                }
            ]
        elif (self.prompt_version == 5):
            input = [
                {"role": "system",
                 "content": "You are computer vision model. Return simple and precise answers."},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text",
                         "text": f"What age is person in bounding box {bbox}? Result should be in either adult (for person older than 14) or minor (for person younger than 14). If there is no person in the image, return unknown."},

                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_patch}",
                        },
                    ],
                }
            ]

        return self.client.responses.create(
            model=self.llm_model_name,
            timeout=600,
            input=input,
        )

    def get_name(self):
        return 'llm_' + ('p' + str(self.prompt_version) + '_' if self.prompt_version>0 else '') + self.llm_model_name

IMAGE_SUFFIXES = ['.jpg', '.jpeg', '.png']


def process_patch(engine, image, bbox):

    response = engine(image, bbox)
    raw_response = response.output_text.lower()
    print(raw_response)
    if('adult' in raw_response):
        label = 'adult'
    elif('minor' in raw_response or 'child' in raw_response or 'teen' in raw_response or 'young' in raw_response or 'baby' in raw_response):
        label = 'minor'
    else:
        label = 'unknown'

    return label, raw_response


def estimate_age_from_patches(
                ae_method, ae_method_name, filename, imgfilename,
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
                                label,ae_text = ae_method(img, detentry['bbox'])
                                if(label is not None):
                                    fixedres['ae_text'] = ae_text

                                    if(label=='minor'):
                                        minors_patches += 1
                                    elif(label=='adult'):
                                        adults_patches += 1
                                    fixedres['label'] = label
                                    estimated_patches += 1
                                else:
                                    fixedres['label'] = 'unknown'

                                fixed_data.append(fixedres)


                    with open(filename.replace( suffix, '_with_age_'+ae_method_name+'.json'), 'w') as outf:
                        json.dump(fixed_data, outf, indent=2)

    except Exception as e:
        print("Error processing file {}: {}".format(filename,e))
    return correct_detection_jsons, all_patches, estimated_patches, adults_patches, minors_patches



def run_age_estimation(imagesdir, outputdir, detection_engine_name, port=8010,
    suffix = "_all_detections.json"):

    ae_engine = LLMEstimator(port)
    ae_method_name = ae_engine.get_name()


    ae_method = lambda image, bbox: process_patch(ae_engine, image, bbox)


    root_outputdir = Path(outputdir,detection_engine_name)

    correct_detection_jsons=0
    all_patches=0
    estimated_patches=0
    adults_patches=0
    minors_patches=0
    estimate_age_total_s = 0.0
    estimate_age_calls = 0
    for filename in root_outputdir.rglob('*'+suffix):
        if filename.is_file():
            if(os.path.exists(str(filename).replace(suffix, '_with_age_'+ae_method_name+'.json'))):
                data = json.load(open(str(filename).replace(suffix, '_with_age_'+ae_method_name+'.json'), 'r'))
                correct_detection_jsons += 1
                all_patches += len(data)
                estimated_patches += len([patch for patch in data if patch['label'] != 'unknown'])
                adults_patches += len([patch for patch in data if patch['label'] == 'adult'])
                minors_patches += len([patch for patch in data if patch['label'] == 'minor'])
            else:
                relpath = os.path.split(os.path.relpath(filename, root_outputdir))[-2]
                stem = os.path.basename(str(filename))[: -len(suffix)]
                imgfilename = os.path.join(imagesdir, relpath, stem + ".jpg")
                t0 = time.perf_counter()
                correct_detection_jsons, all_patches, estimated_patches, adults_patches, minors_patches = (
                    estimate_age_from_patches(ae_method, ae_method_name, str(filename), imgfilename, correct_detection_jsons, all_patches, estimated_patches, adults_patches, minors_patches, suffix))
                estimate_age_total_s += (time.perf_counter() - t0)
                estimate_age_calls += 1

    avg_s = (estimate_age_total_s / estimate_age_calls) if estimate_age_calls else 0.0
    summary = (
        f"Correct entries JSONS in {correct_detection_jsons} results. "
        f"For {estimated_patches}/{all_patches} bboxes age is estimated. "
        f"{adults_patches} are adults, {minors_patches} are minors. "
        f"{all_patches-adults_patches-minors_patches} are uknown. "
        f"Avg processing time: {avg_s:.4f}s over {estimate_age_calls} files."
    )
    print(summary)
    summary_path = os.path.join(root_outputdir, "summary_ae_"+ae_method_name+".txt")
    with open(summary_path, "w", encoding="utf-8") as outf:
        outf.write(summary + "\n")



if __name__ == '__main__':


    outputdir = os.environ.get('OUTPUTDIR', '')
    imagesdir = os.environ.get('IMAGESDIR', '')

    detection_engine_name = 'detector_yolo26x'

    PORT = 8010

    run_age_estimation(imagesdir, outputdir, detection_engine_name, port=PORT)



