import json
import os
import base64

notebooks = ['ej1.ipynb', 'ej2.ipynb', 'ej3.ipynb', 'ej4.ipynb', 'ej5.ipynb']
out_img_dir = os.path.join('informe', 'imagenes_informe')
os.makedirs(out_img_dir, exist_ok=True)

with open('extract_output.txt', 'w', encoding='utf-8') as out_txt:
    for nb in notebooks:
        out_txt.write(f"--- Processing {nb} ---\n")
        if not os.path.exists(nb):
            out_txt.write(f"{nb} not found.\n")
            continue
        with open(nb, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img_counter = 1
        for i, cell in enumerate(data.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source = "".join(cell.get('source', []))
                for output in cell.get('outputs', []):
                    if 'data' in output:
                        if 'image/png' in output['data']:
                            img_data = output['data']['image/png']
                            img_name = f"{nb.split('.')[0]}_img_{img_counter}.png"
                            img_path = os.path.join(out_img_dir, img_name)
                            with open(img_path, 'wb') as img_f:
                                img_f.write(base64.b64decode(img_data))
                            out_txt.write(f"  Saved {img_path} (from cell with source: {source[:50]}...)\n")
                            img_counter += 1
                        
                        if 'text/plain' in output['data']:
                            text_data = "".join(output['data']['text/plain'])
                            if any(k in text_data for k in ['precision', 'accuracy', 'Fairness', 'Parity', 'Threshold', 'TP:', 'Disparate']):
                                out_txt.write(f"  Text output from cell {i}:\n{text_data}\n")
                    
                    if output.get('name') == 'stdout':
                        text_data = "".join(output.get('text', []))
                        if any(k in text_data for k in ['precision', 'accuracy', 'Fairness', 'Parity', 'Threshold', 'TP:', 'Disparate', '==']):
                            out_txt.write(f"  Stdout output from cell {i}:\n{text_data}\n")
