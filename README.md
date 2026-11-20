# MyMaizeCare — Compact Dynamic UI + XAI + Local LLM

## Run the app (Windows PowerShell)

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

You can also double-click `run_app.bat` after dependencies are installed.

## Local LLM without an API key

MyMaizeCare uses Ollama for local report generation. No OpenAI API key is needed.

1. Install Ollama for Windows from https://ollama.com/download
2. Open Ollama and keep it running.
3. In PowerShell, download a compact multilingual model:

   ```powershell
   ollama pull qwen2.5:3b
   ```

4. Start MyMaizeCare:

   ```powershell
   python -m streamlit run app.py
   ```

5. In Diagnosis, open **Local LLM settings** if you want to change the model name. The default is `qwen2.5:3b`.
6. Click **Generate Local AI Report**.

Ollama runs on `http://localhost:11434` by default. You can change the host by setting `OLLAMA_HOST`.

### Important deployment note

This local Ollama setup works when Streamlit and Ollama run on the same computer. If MyMaizeCare is later deployed to a public cloud server, that server needs its own Ollama installation and suitable CPU/RAM/GPU. A user's local Ollama is not automatically accessible to a remote web server.

## Features

- Compact, more dynamic dashboard styling
- Seven maize-leaf classes
- Image upload and camera capture
- RiceLeaf-DMNet v2 checkpoint classification
- Grad-CAM, Grad-CAM++, and LIME
- Heatmap and overlay visualizations
- Local LLM report using Ollama (no paid API key)

## Model

- `Models/RiceLeaf-DMNet.py` — supplied architecture
- `model/best_model.pth` — fine-tuned checkpoint
- Input: 224 × 224
- Normalization: ImageNet

The training transform implementation was not included in the supplied files, so the current resize/normalization should be checked against the original training pipeline for exact parity.

## XAI note

XAI maps show image regions that influenced the model. They do not prove the biological cause of a symptom.

## Advisory safety

The report is an AI screening aid, not a confirmed agronomic diagnosis. It avoids unsupported pesticide rates and recommends checking local agricultural extension guidance and approved product labels.


## Home-screen class symptom descriptions

Each of the seven class cards now includes a short "Typical symptoms" section:
- Fall Armyworm: irregular holes, ragged edges, windowing and possible frass.
- Healthy Leaf: even green colour and no obvious visible damage.
- Leaf Spot: brown/tan/grey spots that may enlarge or merge.
- Magnesium Deficiency: interveinal yellowing, usually on older leaves.
- Nitrogen Deficiency: older-leaf yellowing, often in a V-shape from the tip.
- Northern Leaf Blight: long cigar-shaped lesions along the leaf.
- Phosphorus Deficiency: stunting and dark-green to purplish older leaves.

These are general visual indicators. Similar symptoms can have different causes, so the descriptions are educational guidance rather than confirmed diagnoses.


## Latest home-screen and dark-mode update

- Dark mode toggle is rendered before theme CSS is generated, so the selected theme is applied on the same Streamlit rerun.
- Dark colours are applied to the app shell, cards, expanders, file uploader, camera area, select inputs, and text controls.
- Home-screen class images are displayed as fixed landscape previews using in-memory thumbnails. Original files in `assets/disease_images/` are not modified.
- Class symptom descriptions remain visible below each class description.
