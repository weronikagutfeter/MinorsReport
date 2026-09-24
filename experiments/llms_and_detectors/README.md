# Classical Object Detectors vs. MLLMs for Child Detection in Images

## The task
The main goal of this experiment is to identify all children appearing in the images from the [Parenting Slop Dataset](../../data/parenting_slop_dataset/README.md). (Links and broader description of data can be found in **data/parenting_slop_dataset** directory). 
A key assumption is to minimize or eliminate the need for additional data collection for training and do not use cloud-based infrastructure.
Therefore, all models considered in the comparison are open-source and open-weighted.

## Methodology
The following no-finetune approaches are explored:

1. **Classical person detector + age estimation:** A conventional object detector is first used to identify all people in the image. An age estimation model is then applied to the detected regions to distinguish children from adults and filter out the latter.
1. **MLLM-based child localization:** An MLLM is prompted to identify and localize all children visible in the image, providing a direct end-to-end approach to child detection.
1. **Hybrid solutions** Combining the power of the classic detectors and MLLMs

## Workflow 1: Classical person detector + age estimation

| Step | Script | Role |
|------|--------|------|
| Run person detection | `run_detection_with_yolo.py`, `run_detection_with_mm.py`, or `run_detection_with_roboflow.py` | Detects people (or child/adult classes) and writes detection JSON |
| Estimate age | `../age_estimation/run_estimation_with_mivolo.py`, `run_estimation_with_insightface.py`, or `run_estimation_with_llm.py` | Scores detected person crops and filters adults to keep minors |
| Visualize | `draw_boxes_from_detector_json.py` | Draws predicted boxes on frames for qualitative review |
| Evaluate | `evaluate_coco_detection.py` | Scores predictions against COCO-format GT |

## Workflow 2: MLLM-based child localization

See [Running custom LLMs with llama.cpp](llama_helper.md) to start a local vision model before detection.

| Step | Script | Role |
|------|--------|------|
| Serve MLLM | `llama-server` (llama.cpp) | Hosts a local OpenAI-compatible vision API (port `8010`) |
| Run LLM detection | `run_detection_with_llm.py` | Sends frames to the local vision LLM and saves raw text responses |
| Parse LLM output | `llm_result_to_jsons.py` | Converts raw LLM dumps into structured detection JSON |
| Visualize | `draw_boxes_from_detector_json.py` | Draws predicted boxes on frames for qualitative review |
| Evaluate | `evaluate_coco_detection.py` | Scores predictions against COCO-format GT |


## Results

The detection results were evaluated using the standard COCO detection metrics: AP (Mean Average Precision), AP@0.5 (Average Precision at IoU=0.50, AP@0.75 (Average Precision at IoU=0.75 and AR (Average Recall with a maximum of 100 detections per image).

Baseline detector in the comparison was YOLO26x from [Ultralytics](https://docs.ultralytics.com/models/yolo26). For end-to-end pipeline with MLLMs two prompts was used:
all-people prompt - model was asked to localize all the people in image and attach them a label adult or child, only-children propmpt - model was asked directly to localize only children.


| Approach                                            | Details                                   | AP | AP@0.5  | AP@0.75 | AR | 
|-----------------------------------------------------|-------------------------------------------|----|---------|-----|----|
| Baseline person detector                            | YOLO26x + no age estimation               | 0.546| 0.672   |0.599|**0.772**|
| Detector + Age estimation with classification model | YOLO26x + InsightFace                     | 0.223| 0.263   |0.245|0.302|
| Detector + Age estimation with classification model | YOLO26x + MiVOLO                          | 0.428| 0.500   |0.451|0.454|
| End-to-end MLLM                                     | Qwen3.5-4B _(all-people prompt)_          | 0.401| 0.726   |0.399|0.533|
| End-to-end MLLM                                     | Qwen3.5-4B _(children-only prompt)_       | 0.417| 	0.733  |	0.413|	0.567|
| End-to-end MLLM                                     | Qwen3.6-27B   _(all-people prompt)_       |0.618	| 0.822   |0.693	| 0.701 |
| End-to-end MLLM                                     | Qwen3.6-27B   _(children-only prompt)_    |0.669	| **0.897**	  | **0.758**	| **0.772**|
| End-to-end MLLM                                     | Qwen3.6-35B-A3B  _(all-people prompt)_    | 0.619| 0.828   |0.705|0.756|
| End-to-end MLLM                                     | Qwen3.6-35B-A3B  _(children-only prompt)_ | 0.585 | 	0.829	 | 0.645	| 0.734|
| Detector + Age estimation with MLLM                 | YOLO26x + Qwen3.5-4B        | 0.675| 0.809 |0.734|0.752|
| Detector + Age estimation with MLLM                 | YOLO26x + Qwen3.6-27B       |**0.682**| 0.810 | 0.743 | 0.734     |
| Detector + Age estimation with MLLM                 | YOLO26x + Qwen3.6-35B-A3B   | 0.593| 0.701 |0.638|0.628|
