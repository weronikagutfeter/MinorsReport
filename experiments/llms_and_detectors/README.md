# Classical Object Detectors vs. MLLMs for Child Detection in Images

## The task
The main goal of this experiment is to identify all children appearing in the images from the Parenting Slop Dataset. 
A key assumption is to minimize or eliminate the need for additional data collection and do not use cloud-based infrastructure.
All modeles considered in the comparison are open-source and open-weighted.

## Methodology
The following no-train approaches are explored:

1. **Classical person detector + age estimation:** A conventional object detector is first used to identify all people in the image. An age estimation model is then applied to the detected regions to distinguish children from adults and filter out the latter.
1. **MLLM-based child localization:** An MLLM is prompted to identify and localize all children visible in the image, providing a direct end-to-end approach to child detection.
1. **Hybrid solutions** Combining the power of the classic detectors and MLLMs

## Workflow 1: Person detector + age estimation

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

| Approach | Details | AP | AP@0.5 | AP@0.75 | AR | Prec. | Rec. | 
|----------|---------|----|--------|--------|----|-------|------|
| Detector + Age estimation with classification model | YOLO26x + MiVOLO |  0.428 | 0.500 | 0.451 | 0.454 | 0.401 | 0.511 |
| End-to-end MLLM | Qwen3.6-35B-A3B  | 0.613 | 0.824 | 0.694 |0.744 | 0.899 | 0.891 |
| Detector + Age estimation with MLLM | YOLO26x + Qwen3.6-27B | 0.679 | 0.814 | 0.739 | 0.735 | 0.660 | 0.842 |
