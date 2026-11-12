"""Kaggle GPU notebook: embed the Curanex library (backend/data/knowledge/embed_input.jsonl) with all-MiniLM-L6-v2.
Upload embed_input.jsonl as the private dataset anshu004/curanex-library-input first.
Run from the repo root: python kaggle/make_library_embed_notebook.py
Output: embeddings.npy (float16, L2-normalised, same order as the input) + embedding_ids.json -> copy into backend/data/knowledge/.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTRO = """# Curanex library: embed passages (GPU)

Same model and settings the app uses for search (sentence-transformers all-MiniLM-L6-v2, L2-normalised), so query and passage
vectors match. Order and ids are kept exactly as in embed_input.jsonl."""
CODE = r'''import glob, json
import numpy as np
from sentence_transformers import SentenceTransformer
path = glob.glob("/kaggle/input/**/embed_input.jsonl", recursive=True)[0]
items = [json.loads(l) for l in open(path, encoding="utf-8")]
print("passages:", len(items))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
vecs = model.encode([i["text"] for i in items], batch_size=256, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
np.save("/kaggle/working/embeddings.npy", vecs.astype(np.float16))
json.dump([i["id"] for i in items], open("/kaggle/working/embedding_ids.json", "w"))
print("done", vecs.shape)'''
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": INTRO},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": "!pip install -q sentence-transformers"},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": CODE}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = HERE / "run_library_embed"
d.mkdir(exist_ok=True)
(d / "library_embed.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "anshu004/curanex-library-embed", "title": "Curanex library embed", "code_file": "library_embed.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": ["anshu004/curanex-library-input"], "competition_sources": [], "kernel_sources": []}, indent=1))
print("written", d)
