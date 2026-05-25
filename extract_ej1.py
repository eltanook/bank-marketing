import json
import os
import base64
import re

out_img_dir = os.path.join('informe', 'imagenes_informe')
os.makedirs(out_img_dir, exist_ok=True)

with open('ej1.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

for c in data.get('cells', []):
    source = "".join(c.get('source', []))
    matches = re.findall(r'\[(image\d+)\]:\s*<data:image/png;base64,(.*?)>', source)
    for name, b64 in matches:
        img_name = f"ej1_{name}.png"
        img_path = os.path.join(out_img_dir, img_name)
        with open(img_path, 'wb') as f:
            f.write(base64.b64decode(b64))
        print(f"Saved {img_path}")
