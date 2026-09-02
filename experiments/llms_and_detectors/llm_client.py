from io import BytesIO
from PIL import Image
import base64
from openai import OpenAI



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
                }
            ],
        )

    def get_name(self):
        return 'llm_' + self.llm_model_name


