import json
import os
import base64

output_dir = r'c:\FACULTAD\Equidad en AA\TP_equidades\presentacion\imagenes_para_las_diapos'
os.makedirs(output_dir, exist_ok=True)

files = ['ej1.ipynb', 'ej2.ipynb', 'ej3.ipynb', 'ej4.ipynb', 'ej5.ipynb']
summary = {}

for f in files:
    if not os.path.exists(f):
        continue
    with open(f, 'r', encoding='utf-8') as fh:
        nb = json.load(fh)
    
    img_count = 0
    headers = []
    
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'markdown':
            lines = cell.get('source', [])
            for line in lines:
                if line.startswith('#'):
                    headers.append(line.strip())
        elif cell['cell_type'] == 'code':
            for out in cell.get('outputs', []):
                if 'data' in out and 'image/png' in out['data']:
                    img_data = out['data']['image/png']
                    img_count += 1
                    img_filename = f'{f.replace(".ipynb", "")}_img{img_count}.png'
                    img_path = os.path.join(output_dir, img_filename)
                    with open(img_path, 'wb') as img_file:
                        img_file.write(base64.b64decode(img_data))
    
    summary[f] = {'headers': headers, 'images_extracted': img_count}

print(json.dumps(summary, indent=2))
