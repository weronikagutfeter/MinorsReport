import os
from pathlib import Path
import time
from llm_client import LLMDetector

IMAGE_SUFFIXES = ['.jpg', '.jpeg', '.png']


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



if __name__ == '__main__':

    PORT = 8010
    ONLY_CHILDREN = True


    outputdir = os.environ.get('OUTPUTDIR', '')
    imagesdir = os.environ.get('IMAGESDIR', '')

    engine = LLMDetector(PORT, ONLY_CHILDREN)

    root_inputdir = Path(imagesdir)
    root_outputdir = Path(outputdir,engine.get_name()+('_chonly' if ONLY_CHILDREN else ''))

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
            processed_results = process_filename(engine,str(filename), suboutputdir, processed_results)
            process_filename_total_s += (time.perf_counter() - t0)
            process_filename_calls += 1

    avg_s = (process_filename_total_s / process_filename_calls) if process_filename_calls else 0.0
    summary = f"Processed files: {processed_results}. Avg processing time: {avg_s:.4f}s over {process_filename_calls} files."
    print(summary)
    summary_path = os.path.join(root_outputdir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as outf:
        outf.write(summary + "\n")



