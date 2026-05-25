import nbformat
from nbclient import NotebookClient

with open('ej1.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()

with open('ej1.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
