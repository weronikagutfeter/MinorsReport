import os
from pathlib import Path
import time
from io import BytesIO
from PIL import Image
import base64
from openai import OpenAI

IMAGE_SUFFIXES = ['.jpg', '.jpeg', '.png']




def image_to_base64(filename, mime: str = "image/jpeg", quality: int = 90) -> str:

    if mime not in {"image/jpeg", "image/png"}:
        raise ValueError(f"Unsupported mime: {mime}")

    img = Image.open(filename)
    buf = BytesIO()
    if mime == "image/jpeg":
        img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    else:
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class LLMDetector(object):
    llm_model_name = None
    def __init__(self, port, only_children=False):

        self.port = port
        self.only_children = only_children
        self.client = OpenAI(
            base_url="http://127.0.0.1:{}/v1".format(self.port),
            api_key="sk-no-key-required"
        )
        models = self.client.models.list().data
        if(len(models)>0):
            self.llm_model_name = models[0].id.replace('-UD-Q4_K_XL.gguf', '')
        else:
            raise ValueError("No models found")

    def __call__(self, filename):

        base64_image = image_to_base64(filename)
        return self.client.responses.create(
            model=self.llm_model_name,
            timeout=600,
            input=[

                {"role": "system",
                 "content": "You are object detector that returns results in JSON format"},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text",
                         "text": "Detect all the "+("children" if self.only_children else "persons")+" in the image and return their locations and descriptions in the form of coordinates. Results should be in JSON format: { 'user': {'bbox': [x1, y1, x2, y2], 'description': 'description of the person', 'label':'adult|child' } }"},

                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                },
            ],
        )

    def get_name(self):
        return 'llm_' + self.llm_model_name


def process_filename(engine, filename, outputdir=None, processed_results=0):

    print(f"Processing file: {filename}")
    response = engine(filename)
    print(response.output_text)
    if(outputdir is not None):
        raw_text = response.output_text
        with open(os.path.join(outputdir, os.path.basename(filename).replace('.jpg', '_raw.txt')), 'w') as outf:
            outf.write(raw_text)
            processed_results += 1

    return processed_results

def run_detection(imagesdir, outputdir, port, only_children=False):
    engine = LLMDetector(port, only_children)

    root_inputdir = Path(imagesdir)
    root_outputdir = Path(outputdir, engine.get_name() + ('_chonly' if only_children else ''))

    processed_results = 0
    process_filename_total_s = 0.0
    process_filename_calls = 0
    for filename in root_inputdir.rglob('*'):
        if filename.is_file() and filename.suffix.lower() in IMAGE_SUFFIXES:
            relpath = os.path.split(os.path.relpath(filename, root_inputdir))[-2]
            suboutputdir = os.path.join(root_outputdir, relpath)
            if not os.path.exists(suboutputdir):
                os.makedirs(suboutputdir)

            t0 = time.perf_counter()
            processed_results = process_filename(engine, str(filename), suboutputdir, processed_results)
            process_filename_total_s += (time.perf_counter() - t0)
            process_filename_calls += 1

    avg_s = (process_filename_total_s / process_filename_calls) if process_filename_calls else 0.0
    summary = f"Processed files: {processed_results}. Avg processing time: {avg_s:.4f}s over {process_filename_calls} files."
    print(summary)
    summary_path = os.path.join(root_outputdir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as outf:
        outf.write(summary + "\n")


if __name__ == '__main__':

    PORT = 8010
    ONLY_CHILDREN = True


    outputdir = os.environ.get('OUTPUTDIR', '')
    imagesdir = os.environ.get('IMAGESDIR', '')

    run_detection(imagesdir, outputdir, PORT, only_children=ONLY_CHILDREN)




