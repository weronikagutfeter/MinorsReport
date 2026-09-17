import json
from pathlib import Path
import os
from PIL import Image
import re


def simple_validation(data):

    if(isinstance(data, list)):
        correct = True
        for item in data:
            if(not isinstance(item, dict) or 'bbox' not in item or 'description' not in item or 'label' not in item):
                correct=False
                break
    elif(isinstance(data, dict)):
        correct = 'bbox' in data and 'description' in data and 'label' in data
    else:
        correct=False
    return correct

def trim_llm(raw_text):
    jsontag = raw_text.find('```json')
    if(jsontag>=0):
        raw_text = raw_text[jsontag+8:]
    closingtag = raw_text.find('```')
    if(closingtag>=0):
        raw_text = raw_text[:closingtag]

    raw_text = re.sub(r"[\n\t\r]*", "", raw_text)

    sidx1 = raw_text.find('[')
    sidx2 = raw_text.find('{')
    if(sidx2<0 or sidx1<sidx2):
        sidx = sidx1
        closingbracket = ']'
    else:
        sidx = sidx2
        closingbracket = '}'
    processed_text = None
    if(sidx>=0):
        processed_text = raw_text[sidx:]
        eidx = processed_text.rfind(closingbracket)
        if(eidx>=1):
            processed_text = processed_text[:eidx+1]
    return processed_text

def fixentry(engine_name, detresult, width, height):
    fixedres = {}
    errflag = False
    bbox_key = next((k for k in detresult if k.startswith("bbox")), None)
    if(bbox_key!='bbox'):
        errflag = True
    if bbox_key is not None:
        x1, y1, x2, y2 = (int(round(float(v))) for v in detresult[bbox_key])
        if ('gemma' in engine_name.lower()):
            # new_x1 = int(y1 / 896 * width)
            # y1 = int(x1 / 896 * height)
            # new_x2 = int(y2 / 896 * width)
            # y2 = int(x2 / 896 * height)
            new_x1 = int(y1 / 1000 * width)
            y1 = int(x1 / 1000 * height)
            new_x2 = int(y2 / 1000 * width)
            y2 = int(x2 / 1000 * height)
            x1 = new_x1
            x2 = new_x2

        else:
            y1 = int(y1 / 1000 * height)
            x1 = int(x1 / 1000 * width)
            y2 = int(y2 / 1000 * height)
            x2 = int(x2 / 1000 * width)
        if(x2<x1):
            x2=x1
            errflag = True
        elif(y2<y1):
            y2=y1
            errflag = True
    else:
        x1, y1, x2, y2 = [0, 0, 0, 0]
        errflag = True



    fixedres["bbox"] = [x1, y1, x2, y2]

    dkey = next((k for k in detresult if k.startswith("descr")), None)
    if dkey is not None:
        description = detresult[dkey]
    else:
        description = ""
        errflag = True

    lkey = next((k for k in detresult if k.startswith("label")), None)
    if lkey is not None:
        label = detresult[lkey].lower()
    else:
        label = ''
        errflag = True
    if (label != 'adult' and label != 'child'):
        errflag = True
        description = description.lower()
        if(description==''):
            description = label.lower()
        if ('child' in description):
            label = 'child'
        elif ('baby' in description):
            label = 'child'
        elif ('girl' in description):
            label = 'child'
        elif ('boy' in description):
            label = 'child'
        elif ('adult' in description):
            label = 'adult'
        elif ('person' in description):
            label = 'adult'
        elif ('man' in description):
            label = 'adult'
        elif ('woman' in description):
            label = 'adult'
        else:
            label = 'unknown'

    fixedres["label"] = label
    fixedres["description"] = description
    return fixedres, errflag

def extract_json_from_llm(engine_name, filename, imgfilename, correct_jsons, processed_results, suffix):
    fixed_data = []

    try:
        raw_data = ''
        with Image.open(imgfilename) as im:
            width, height = im.size

        with open(filename, encoding="utf-8") as f:
            raw_data = f.read()
            data = trim_llm(raw_data)
            noresultstag = False
            if(data is None):
                lowerdata = raw_data.lower()
                if('no person' in lowerdata or 'any person' in lowerdata):
                    data="[]"
                    noresultstag = True
            data = json.loads(data)
            if(data is not None):
                if(not(noresultstag) and simple_validation(data)):
                    correct_jsons += 1
                #normalize coordinates and fix corrupted files
                if isinstance(data, dict):
                    if len(data)>0:
                        if 'user' in data:
                            data = data['user']
                        elif 'users' in data:
                            data = data['users']
                        elif 'person' in data:
                            data = data['person']
                        else:
                            data = list(data.values())[0]
                    else:
                        data = []

                if not isinstance(data, list):
                    data = [data]
                for detresult in data:
                    if('label' in detresult or 'description' in detresult):
                        fixedres, errflag = fixentry(engine_name, detresult, width, height)
                    elif isinstance(detresult, dict) and len(detresult)>0:
                        entrykey = next(iter(detresult))
                        fixedres, errflag = fixentry(engine_name, detresult[entrykey], width, height)
                    fixed_data.append(fixedres)

                processed_results += 1
                with open(filename.replace( suffix, '_fixed.json'), 'w') as outf:
                    json.dump(fixed_data, outf, indent=2)

    except Exception as e:
        print("Error processing file {}: {}\n {}".format(filename,e,  raw_data))
    return correct_jsons, processed_results


def run_postprocessing(imagesdir, outputdir,llm_engine_name, suffix= "_raw.txt"):
    root_outputdir = Path(outputdir, llm_engine_name)

    correct_jsons = 0
    processed_results = 0
    for filename in root_outputdir.rglob('*' + suffix):
        if filename.is_file():
            relpath = os.path.split(os.path.relpath(filename, root_outputdir))[-2]
            stem = os.path.basename(str(filename))[: -len(suffix)]
            imgfilename = os.path.join(imagesdir, relpath, stem + ".jpg")
            correct_jsons, processed_results = extract_json_from_llm(
                llm_engine_name, str(filename), imgfilename, correct_jsons, processed_results, suffix
            )
    summary = f"Correct entries JSONS in {correct_jsons} results. {processed_results} are fixable"
    print(summary)
    summary_path = os.path.join(root_outputdir, "summary_fixing.txt")
    with open(summary_path, "w", encoding="utf-8") as outf:
        outf.write(summary + "\n")


if __name__ == '__main__':

    SUFFIX = "_raw.txt"

    outputdir = os.environ.get('OUTPUTDIR', '')
    imagesdir = os.environ.get('IMAGESDIR', '')

    engine = 'llm_gemma-3-4b-it'
    engine = 'llm_gemma-4-E4B-it'
    # engine =  'llm_Qwen3.5-4B'
    # engine = 'llm_Qwen3.6-35B-A3B'
    # engine = 'llm_Qwen3.6-35B-A3B_chonly'
    # engine = 'llm_Qwen3.5-4B_chonly'
    engine = 'llm_Qwen3.6-27B'

    run_postprocessing(imagesdir, outputdir, engine, SUFFIX)







