import json
import os
import time
from pathlib import Path

from ultralytics import YOLO, YOLOE

IMAGE_SUFFIXES = ['.jpg', '.jpeg', '.png']
CONF_TH = 0

class YoloDetector(object):
    def __init__(self, weights):

        self.weights = weights
        self.model = YOLO(self.weights)
        self.key_class_name = 'person'
        if('yoloe' in self.weights or 'world' in self.weights):
            self.model.set_classes(['child','adult'])
            self.key_class_name = 'child'



    def __call__(self, img):
        return self.model(img, verbose=False)


    def get_name(self):
        return 'detector_'+os.path.basename(self.weights).split('.')[0]


def detection_results_to_json(xyxy, names, confs):

    detections = []
    n = len(names)
    for i in range(n):
        box = xyxy[i]
        detections.append(
            {
                "class": names[i],
                "confidence": float(confs[i]),
                "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
            }
        )
    return json.dumps(detections, indent=2)


def process_filename(engine, filename, outputdir=None, raw_results=0, processed_results=0):

    print(f"Processing file: {filename}")

    results = engine(filename)
    for result in results:
        xyxy = result.boxes.xyxy.cpu().numpy()  # top-left-x, top-left-y, bottom-right-x, bottom-right-y

        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf.cpu().numpy()  # confidence score of each box

        ids = list(filter(lambda x: names[x] == engine.key_class_name and confs[x]>CONF_TH, list(range(len(names)))))
        print(f"Detected {len(ids)} people")

        if (outputdir is not None):
            raw_text = detection_results_to_json(xyxy, names, confs)
            with open(os.path.join(outputdir, os.path.basename(filename).replace('.jpg', '_all_detections.json')), 'w') as outf:
                outf.write(raw_text)
                raw_results += 1
            xyxy = xyxy[ids]
            names = [names[id] for id in ids]
            confs = confs[ids]
            processed_text  = detection_results_to_json(xyxy, names, confs)
            if (processed_text is not None):
                with open(os.path.join(outputdir, os.path.basename(filename).replace('.jpg', '_fixed.json')),
                          'w') as outf:
                    outf.write(processed_text)
                    processed_results += 1



    return raw_results,processed_results

if __name__ == '__main__':

    outputdir = os.environ.get('OUTPUTDIR', '')
    imagesdir = os.environ.get('IMAGESDIR', '')
    modelsdir = os.environ.get('MODELSDIR', '')


    # modelname= 'yolo26x.pt'
    # modelname = 'yolo11x.pt'
    # modelname ='yolov3u.pt'
    modelname = 'yoloe-26x-seg.pt'
    # modelname = 'yolov8x-world.pt'


    weights = os.path.join(modelsdir,modelname)
    engine = YoloDetector(weights)

    root_inputdir = Path(imagesdir)
    root_outputdir = Path(outputdir,engine.get_name())

    raw_results = 0
    processed_results = 0
    process_filename_total_s = 0.0
    process_filename_calls = 0
    for filename in root_inputdir.rglob('*'):
        if filename.is_file() and filename.suffix.lower() in IMAGE_SUFFIXES:
            relpath = os.path.split(os.path.relpath(filename, root_inputdir))[-2]
            suboutputdir = os.path.join(root_outputdir,relpath)
            if not os.path.exists(suboutputdir):
                os.makedirs(suboutputdir)

            t0 = time.perf_counter()
            raw_results,processed_results = process_filename(engine,str(filename), outputdir=suboutputdir, raw_results=raw_results, processed_results=processed_results)
            process_filename_total_s += (time.perf_counter() - t0)
            process_filename_calls += 1

    avg_s = (process_filename_total_s / process_filename_calls) if process_filename_calls else 0.0
    summary = f"Correct JSONS in {processed_results}/{raw_results} results. Avg processing time: {avg_s:.4f}s over {process_filename_calls} files."
    print(summary)
    summary_path = os.path.join(root_outputdir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as outf:
        outf.write(summary + "\n")



