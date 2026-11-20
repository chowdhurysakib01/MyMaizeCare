from pathlib import Path
import hashlib
import importlib.util
import io
import os
import sys

import requests

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

st.set_page_config(
    page_title="MyMaizeCare",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "assets" / "disease_images"
CHECKPOINT_PATH = BASE_DIR / "model" / "best_model.pth"
ARCHITECTURE_PATH = BASE_DIR / "Models" / "RiceLeaf-DMNet.py"

CLASS_NAMES = [
    "Fall Armyworm (Insect) Damage",
    "Healthy Leaf",
    "Leaf Spot",
    "Magnesium Deficiency",
    "Nitrogen Deficiency",
    "Northern Leaf Blight",
    "Phosphorus Deficiency",
]
CLASS_INFO = [
    "Leaf damage caused by fall armyworm feeding.",
    "A healthy maize leaf without visible disease symptoms.",
    "Leaf spot symptoms on maize.",
    "Possible visual symptoms associated with magnesium deficiency.",
    "Possible visual symptoms associated with nitrogen deficiency.",
    "Northern leaf blight symptoms on maize.",
    "Possible visual symptoms associated with phosphorus deficiency.",
]

# Short, field-observable symptom descriptions shown on the seven home-screen cards.
# These are general visual indicators, not a substitute for field/lab confirmation.
CLASS_SYMPTOMS = [
    "Irregular holes and ragged edges; scraped or window-like patches; sawdust-like frass may be visible near the whorl.",
    "Even green colour and normal leaf growth; no obvious spots, streaks, holes, yellowing, or unusual wilting.",
    "Small brown, tan, or grey spots; spots may have yellow margins and can enlarge or merge into larger dead areas.",
    "Older leaves may show yellowing between veins while veins remain greener; reddish or purplish tones can sometimes appear.",
    "Older leaves often turn pale green or yellow, with a V-shaped yellowing pattern that starts at the leaf tip along the midrib.",
    "Long, cigar-shaped grey-green to tan lesions that extend along the leaf; lesions may join and cause large dead areas.",
    "Plants may be stunted with unusually dark-green leaves; older leaves can develop reddish-purple or purplish colour, especially under stress.",
]

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
if "page" not in st.session_state:
    st.session_state["page"] = "home"

# -------------------- Compact theme --------------------
dark = st.session_state["dark_mode"]
if dark:
    bg, panel, text, muted, border, accent = (
        "#101a16", "#192720", "#eef7f0", "#b7c9bb", "#2b4033", "#8ed6a0"
    )
else:
    bg, panel, text, muted, border, accent = (
        "#f5faf6", "#ffffff", "#183323", "#607568", "#dce9df", "#247a43"
    )

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
    :root {{ color-scheme: {"dark" if dark else "light"}; }}
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stHeader"], [data-testid="stBottom"] {{
        background-color:{bg} !important; color:{text} !important;
        font-family:'DM Sans',sans-serif;
    }}
    [data-testid="stAppViewContainer"] {{
        background-image:
          radial-gradient(ellipse at 8% 5%, rgba(35,150,88,.12), transparent 34%),
          radial-gradient(ellipse at 92% 18%, rgba(39,174,145,.09), transparent 30%) !important;
    }}
    [data-testid="stHeader"] {{ background:transparent !important; }}
    .block-container {{ max-width:1160px; padding-top:.75rem; padding-bottom:1.6rem; }}
    .stMarkdown, .stMarkdown p, .stMarkdown li, label,
    [data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"] {{
        color:{text} !important;
    }}
    h1,h2,h3,h4 {{ font-family:'Manrope','DM Sans',sans-serif !important; color:{text} !important; }}
    h1 {{ font-size:1.75rem !important; margin:.35rem 0 .55rem !important; }}
    h2 {{ font-size:1.32rem !important; margin:.35rem 0 .45rem !important; }}
    h3 {{ font-size:1.08rem !important; margin:.3rem 0 .35rem !important; }}
    p {{ font-size:.91rem; }}
    .brand {{ display:flex; align-items:center; gap:9px; color:{text};
              font-size:1.17rem; font-weight:800; padding:.15rem 0 .55rem;
              letter-spacing:-.035em; }}
    .brand-mark {{ display:inline-flex; width:36px; height:36px; align-items:center;
                   justify-content:center; background:linear-gradient(145deg,#e4f8e9,#bce8c9);
                   border:1px solid rgba(36,122,67,.18); border-radius:12px;
                   font-size:1.15rem; box-shadow:0 5px 18px rgba(20,120,65,.12); }}
    .brand-sub {{ color:{muted}; font-size:.65rem; font-weight:700; letter-spacing:.11em;
                  margin-left:5px; padding:5px 9px; border:1px solid {border};
                  border-radius:999px; }}
    .hero {{ position:relative; overflow:hidden;
             background:linear-gradient(115deg, {panel} 0%, rgba(35,125,75,.13) 58%, {panel} 100%);
             border:1px solid {border}; border-radius:22px;
             padding:1.45rem 1.6rem; margin:.25rem 0 .9rem;
             box-shadow:0 12px 34px rgba(15,70,40,.08); }}
    .hero:after {{ content:""; position:absolute; width:230px; height:230px; right:-55px; top:-95px;
                   border-radius:50%; background:radial-gradient(circle,rgba(63,190,112,.22),transparent 68%);
                   animation:heroPulse 6s ease-in-out infinite; pointer-events:none; }}
    @keyframes heroPulse {{ 0%,100% {{ transform:scale(.92); opacity:.65; }} 50% {{ transform:scale(1.12); opacity:1; }} }}
    .eyebrow {{ color:{accent} !important; font-size:.68rem; font-weight:800;
                letter-spacing:.13em; text-transform:uppercase; }}
    .hero h1 {{ color:{text} !important; font-size:2rem !important;
                line-height:1.12; margin:.38rem 0 .48rem !important; letter-spacing:-.045em; }}
    .hero p {{ color:{muted} !important; font-size:.92rem; margin:0; max-width:650px; }}
    .hero-tag {{ display:inline-flex; align-items:center; gap:6px; margin-top:.9rem;
                 padding:5px 10px; border-radius:999px; color:{accent};
                 background:rgba(40,150,80,.10); border:1px solid rgba(40,150,80,.2);
                 font-size:.7rem; font-weight:700; }}
    .section-heading {{ color:{text} !important; font-size:1.22rem;
                        font-weight:800; margin:.85rem 0 .2rem; letter-spacing:-.025em; }}
    .section-sub {{ color:{muted} !important; font-size:.82rem; margin-bottom:.6rem; }}
    .class-name {{ color:{text} !important; font-weight:750; font-size:.9rem;
                   line-height:1.25; margin:.1rem 0 .3rem; }}
    .class-desc {{ color:{muted} !important; font-size:.78rem; line-height:1.4; }}
    .symptom-title {{ color:{accent} !important; font-size:.66rem; font-weight:800;
                      letter-spacing:.045em; text-transform:uppercase;
                      margin:.42rem 0 .14rem; }}
    .symptom-desc {{ color:{text} !important; font-size:.74rem; line-height:1.38;
                     opacity:.88; margin:0 0 .12rem; }}
    .class-number {{ color:{accent} !important; font-size:.65rem; font-weight:800;
                     letter-spacing:.1em; margin-bottom:.25rem; }}
    .note {{ color:{muted} !important; font-size:.75rem; border-left:3px solid {accent};
             padding:.3rem .6rem; margin:.4rem 0 .7rem; }}
    .result-card {{ background:linear-gradient(135deg,{panel},rgba(40,145,82,.08));
                    border:1px solid {border}; border-radius:15px;
                    padding:.78rem .9rem; min-height:78px;
                    box-shadow:0 6px 20px rgba(10,60,30,.05); }}
    .result-label {{ color:{muted}; font-size:.72rem; margin-bottom:.2rem; }}
    .result-value {{ color:{text}; font-size:1.04rem; font-weight:800; line-height:1.25; }}
    .small-muted {{ color:{muted}; font-size:.75rem; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color:{panel} !important; border-color:{border} !important;
        border-radius:15px !important; transition:transform .2s ease, box-shadow .2s ease;
        box-shadow:0 5px 18px rgba(10,60,30,.035);
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
        transform:translateY(-3px); box-shadow:0 12px 26px rgba(10,90,45,.10);
        border-color:rgba(55,160,95,.45) !important;
    }}
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
        background-color:{panel} !important; border-color:{border} !important; color:{text} !important;
    }}
    div.stButton > button {{
        border-radius:10px !important; padding:.46rem .85rem !important;
        min-height:2.3rem !important; font-size:.86rem !important;
        transition:transform .18s ease, box-shadow .18s ease !important;
    }}
    div.stButton > button:hover {{ transform:translateY(-2px); box-shadow:0 7px 16px rgba(30,130,70,.16); }}
    div.stButton > button[kind="primary"] {{
        background:linear-gradient(105deg,#20834b,#2ea66a) !important;
        border:0 !important; color:white !important; font-weight:750;
    }}
    div[data-testid="stMetric"] {{ padding:.25rem .4rem; }}
    div[data-testid="stMetricValue"] {{ font-size:1.25rem !important; }}
    div[data-testid="stProgress"] {{ margin-top:-.25rem; margin-bottom:.45rem; }}
    [data-testid="stFileUploader"] {{ padding:.2rem; }}
    [data-testid="stAlert"] {{ padding:.55rem .75rem; border-radius:10px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Model --------------------
@st.cache_resource(show_spinner=False)
def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not ARCHITECTURE_PATH.is_file():
        raise FileNotFoundError(f"Architecture file not found: {ARCHITECTURE_PATH}")

    module_name = "mymaizecare_dmnet_arch"
    spec = importlib.util.spec_from_file_location(module_name, ARCHITECTURE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the supplied DMNet architecture.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict.")

    compatibility = checkpoint.get("compatibility", {})
    if compatibility.get("class_names") != CLASS_NAMES:
        raise ValueError("Checkpoint class names/order do not match this app.")
    if compatibility.get("image_size") != 224:
        raise ValueError("Expected 224 × 224 model input in checkpoint metadata.")
    if compatibility.get("normalization") != "imagenet":
        raise ValueError("Expected ImageNet normalization in checkpoint metadata.")

    model = module.build_riceleaf_dmnet(
        num_classes=7, pretrained=False, freeze_backbone=False
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def preprocess_image(image: Image.Image, device):
    image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0).to(device)


def jet_heatmap(cam):
    """Create a compact RGB jet-style heatmap without an extra plotting dependency."""
    x = np.clip(cam.astype(np.float32), 0, 1)
    red = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def make_overlay(original: Image.Image, heatmap: Image.Image, alpha=0.42):
    base = original.convert("RGB").resize(heatmap.size)
    return Image.blend(base, heatmap.convert("RGB"), alpha)


def generate_gradcam(model, input_tensor, target_class, method="Grad-CAM"):
    """Generate Grad-CAM or Grad-CAM++ from the architecture's exposed target layer."""
    target_layer = model.get_target_layer()
    captured = {}

    def save_activation(_module, _inputs, output):
        captured["activation"] = output

    handle = target_layer.register_forward_hook(save_activation)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(input_tensor)
            activation = captured.get("activation")
            if activation is None:
                raise RuntimeError("The Grad-CAM target layer did not produce activations.")
            score = logits[:, target_class].sum()
            gradients = torch.autograd.grad(score, activation, retain_graph=False)[0]

        activations = activation.detach()
        gradients = gradients.detach()

        if method == "Grad-CAM":
            weights = gradients.mean(dim=(2, 3), keepdim=True)
        else:
            # Grad-CAM++ alpha weighting from first-order gradient powers.
            grad_2 = gradients.pow(2)
            grad_3 = gradients.pow(3)
            activation_sum = activations.sum(dim=(2, 3), keepdim=True)
            denominator = 2.0 * grad_2 + activation_sum * grad_3
            alpha = grad_2 / (denominator + 1e-8)
            weights = (alpha * torch.relu(gradients)).sum(
                dim=(2, 3), keepdim=True
            )

        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=(224, 224), mode="bilinear", align_corners=False
        )[0, 0]
        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def generate_lime(model, device, original_image, target_class):
    """Generate a superpixel importance map using LIME."""
    try:
        from lime.lime_image import LimeImageExplainer
    except ImportError as exc:
        raise ImportError(
            "LIME is not installed. Run: python -m pip install lime scikit-image"
        ) from exc

    image_np = np.asarray(
        original_image.convert("RGB").resize((224, 224)), dtype=np.uint8
    )

    def predict_batch(images):
        batch_tensors = []
        for image_arr in images:
            pil_img = Image.fromarray(np.asarray(image_arr).astype(np.uint8)).convert("RGB")
            batch_tensors.append(preprocess_image(pil_img, device)[0])
        batch = torch.stack(batch_tensors).to(device)
        with torch.inference_mode():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    explainer = LimeImageExplainer(random_state=42)
    explanation = explainer.explain_instance(
        image_np,
        classifier_fn=predict_batch,
        labels=(target_class,),
        hide_color=0,
        num_samples=240,
        batch_size=16,
    )

    local_weights = dict(explanation.local_exp[target_class])
    segments = explanation.segments
    score_map = np.zeros(segments.shape, dtype=np.float32)
    for segment_id, weight in local_weights.items():
        score_map[segments == segment_id] = float(weight)

    max_abs = float(np.max(np.abs(score_map))) if score_map.size else 0.0
    if max_abs > 0:
        score_map = score_map / max_abs
    # Display positive evidence as warmer colors and negative evidence as cooler colors.
    heat = np.zeros((*score_map.shape, 3), dtype=np.float32)
    positive = np.clip(score_map, 0, 1)
    negative = np.clip(-score_map, 0, 1)
    heat[..., 0] = positive
    heat[..., 1] = 0.18 * (1 - np.abs(score_map))
    heat[..., 2] = negative
    heatmap = Image.fromarray((np.clip(heat, 0, 1) * 255).astype(np.uint8))
    overlay = Image.blend(Image.fromarray(image_np).convert("RGB"), heatmap, 0.43)
    return heatmap, overlay


def create_local_llm_report(class_name, confidence, probabilities, xai_method, model_name="qwen2.5:3b"):
    """Generate a farmer-friendly report using a locally running Ollama model."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    ranked = sorted(zip(CLASS_NAMES, probabilities), key=lambda pair: pair[1], reverse=True)
    probability_text = "\n".join(
        f"- {name}: {score * 100:.2f}%" for name, score in ranked
    )
    prompt = f"""
You are MyMaizeCare, a careful maize crop screening assistant.
Write a practical, farmer-friendly maize leaf report in clear English.

MODEL OUTPUT
Predicted class: {class_name}
Model score: {confidence:.2f}%
XAI method: {xai_method or "Not generated"}
Class scores:
{probability_text}

REPORT REQUIREMENTS
- Explain the predicted class in plain language.
- List visible symptoms the farmer should inspect on several leaves/plants.
- Give safe, practical next steps for field inspection, crop hygiene, and monitoring.
- Explain prevention relevant to this class.
- If Healthy Leaf is predicted, do not recommend disease treatment; suggest routine monitoring and good crop care.
- This is an AI screening result, not a confirmed diagnosis. A high model score is not a guarantee.
- XAI highlights image regions influencing the model; it does not prove the biological cause.
- Do not invent pesticide names, chemical rates, or application doses. For chemical control, advise contacting a local agricultural extension officer and following locally approved product labels.
- Keep the report concise (about 180-250 words), using short headings and bullet points.
"""
    try:
        response = requests.post(
            f"{host}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.25, "num_predict": 500},
            },
            timeout=240,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Ollama is not running or cannot be reached. Install Ollama, open it, "
            "then run `ollama pull qwen2.5:3b` in PowerShell."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "The local model took too long to respond. Try a smaller Ollama model "
            "or wait for the first model load to finish."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = response.json().get("error", "")
        except Exception:
            pass
        raise RuntimeError(
            f"Ollama returned an error: {detail or response.status_code}. "
            f"Check that model `{model_name}` is downloaded with `ollama pull {model_name}`."
        ) from exc

    data = response.json()
    report = data.get("response", "").strip()
    if not report:
        raise RuntimeError("Ollama returned an empty report. Try generating again.")
    return report


# -------------------- Header --------------------
head_left, head_right = st.columns([5, 2], vertical_alignment="center")
with head_left:
    st.markdown(
        '<div class="brand"><span class="brand-mark">🌽</span><span>MyMaizeCare</span><span class="brand-sub">AI CROP CARE</span></div>',
        unsafe_allow_html=True,
    )
with head_right:
    st.toggle("Dark mode", key="dark_mode")

# -------------------- Home --------------------
if st.session_state["page"] == "home":
    left, right = st.columns([3, 1], vertical_alignment="center", gap="medium")
    with left:
        st.markdown(
            """
            <section class="hero">
                <div class="eyebrow">AI-POWERED MAIZE LEAF CARE</div>
                <h1>Welcome to MyMaizeCare</h1>
                <p>A smarter way to screen maize leaf health, understand model predictions and explore practical crop-care guidance.</p>
                <div class="hero-tag">✦ DMNet AI &nbsp;·&nbsp; Explainable predictions &nbsp;·&nbsp; Local AI reports</div>
            </section>
            """,
            unsafe_allow_html=True,
        )
    with right:
        if st.button("Diagnose Now", type="primary", use_container_width=True):
            st.session_state["page"] = "diagnosis"
            st.rerun()

    st.markdown('<div class="section-heading">Explore the 7 Classes</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Your model classifies maize leaf images into one of these seven classes.</div>',
        unsafe_allow_html=True,
    )
    for start in range(0, 7, 3):
        cols = st.columns(3, gap="small")
        for col, index in zip(cols, range(start, min(start + 3, 7))):
            with col:
                with st.container(border=True):
                    image_path = IMAGE_DIR / f"picture{index + 1}.jpg"
                    if image_path.exists():
                        st.image(str(image_path), use_container_width=True)
                    st.markdown(f'<div class="class-number">CLASS {index + 1:02d}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="class-name">{CLASS_NAMES[index]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="class-desc">{CLASS_INFO[index]}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="symptom-title">Typical symptoms</div>'
                        f'<div class="symptom-desc">{CLASS_SYMPTOMS[index]}</div>',
                        unsafe_allow_html=True,
                    )
    st.markdown(
        '<div class="note">Class images are placeholders. Replace them with your own photos using the same filenames.</div>',
        unsafe_allow_html=True,
    )
    st.caption("MyMaizeCare • Maize Leaf Classification")

# -------------------- Diagnosis --------------------
else:
    nav_col, heading_col = st.columns([1, 5], vertical_alignment="center")
    with nav_col:
        if st.button("← Home"):
            st.session_state["page"] = "home"
            st.rerun()
    with heading_col:
        st.markdown("### Maize Leaf Diagnosis")

    source = st.radio(
        "Image source",
        ["Upload Image", "Capture Image"],
        horizontal=True,
        label_visibility="collapsed",
        key="image_source",
    )
    if source == "Upload Image":
        image_file = st.file_uploader(
            "Choose a maize leaf image",
            type=["jpg", "jpeg", "png"],
            key="uploaded_leaf_image",
        )
    else:
        image_file = st.camera_input("Capture maize leaf", key="captured_leaf_image")

    if image_file is not None:
        image_bytes = image_file.getvalue()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        try:
            original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            original_image = None
            st.error("Could not read this image. Please choose a valid JPG or PNG.")

        if original_image is not None:
            with st.expander("Preview selected image", expanded=False):
                st.image(original_image, use_container_width=True)

            if st.button("Classify Image", type="primary", use_container_width=True):
                try:
                    with st.spinner("Classifying maize leaf..."):
                        model, device = load_model()
                        input_tensor = preprocess_image(original_image, device)
                        with torch.inference_mode():
                            logits = model(input_tensor)
                            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                        pred_idx = int(np.argmax(probs))
                        st.session_state["prediction"] = {
                            "image_hash": image_hash,
                            "predicted_index": pred_idx,
                            "probabilities": probs.tolist(),
                            "device": str(device),
                        }
                        st.session_state.pop("xai_result", None)
                        st.session_state.pop("llm_report", None)
                except Exception as exc:
                    st.error(f"Classification failed: {exc}")

            prediction = st.session_state.get("prediction")
            if prediction and prediction.get("image_hash") == image_hash:
                pred_idx = prediction["predicted_index"]
                probabilities = prediction["probabilities"]
                confidence = probabilities[pred_idx] * 100.0
                predicted_name = CLASS_NAMES[pred_idx]

                st.markdown("#### Classification Result")
                r1, r2 = st.columns([3, 1], gap="small")
                with r1:
                    st.markdown(
                        f'<div class="result-card"><div class="result-label">Predicted class</div>'
                        f'<div class="result-value">{predicted_name}</div></div>',
                        unsafe_allow_html=True,
                    )
                with r2:
                    st.markdown(
                        f'<div class="result-card"><div class="result-label">Model score</div>'
                        f'<div class="result-value">{confidence:.2f}%</div></div>',
                        unsafe_allow_html=True,
                    )
                st.caption(
                    "Model score is not a guaranteed diagnosis or necessarily a calibrated probability."
                )

                ranked = sorted(range(7), key=lambda i: probabilities[i], reverse=True)
                with st.expander("Class probabilities", expanded=False):
                    for idx in ranked:
                        st.write(f"{CLASS_NAMES[idx]} — {probabilities[idx] * 100:.2f}%")
                        st.progress(float(probabilities[idx]))

                # -------------------- XAI --------------------
                st.markdown("#### Explainable AI (XAI)")
                xai_method = st.selectbox(
                    "Choose explanation method",
                    ["Grad-CAM", "Grad-CAM++", "LIME"],
                    key="xai_method",
                )
                if st.button("Generate XAI Explanation", use_container_width=True):
                    try:
                        with st.spinner(f"Generating {xai_method} explanation..."):
                            model, device = load_model()
                            input_tensor = preprocess_image(original_image, device)
                            if xai_method in ("Grad-CAM", "Grad-CAM++"):
                                cam = generate_gradcam(
                                    model, input_tensor, pred_idx, xai_method
                                )
                                heatmap = jet_heatmap(cam).resize((224, 224))
                                resized_original = original_image.resize((224, 224))
                                overlay = make_overlay(resized_original, heatmap)
                            else:
                                heatmap, overlay = generate_lime(
                                    model, device, original_image, pred_idx
                                )
                            st.session_state["xai_result"] = {
                                "image_hash": image_hash,
                                "method": xai_method,
                                "heatmap": heatmap,
                                "overlay": overlay,
                            }
                            st.session_state.pop("llm_report", None)
                    except Exception as exc:
                        st.error(f"XAI generation failed: {exc}")

                xai_result = st.session_state.get("xai_result")
                if xai_result and xai_result.get("image_hash") == image_hash:
                    st.caption(f"Explanation method: {xai_result['method']}")
                    x1, x2 = st.columns(2, gap="small")
                    with x1:
                        st.image(xai_result["heatmap"], caption="Heatmap", use_container_width=True)
                    with x2:
                        st.image(xai_result["overlay"], caption="Overlay", use_container_width=True)
                    st.caption(
                        "Highlighted regions show image areas that influenced the model; they do not prove the biological cause."
                    )

                # -------------------- Local LLM report --------------------
                st.markdown("#### Local AI Report")
                st.caption("Runs on your own computer with Ollama. No OpenAI API key or per-request API charge.")
                with st.expander("Local LLM settings", expanded=False):
                    ollama_model = st.text_input(
                        "Ollama model name",
                        value="qwen2.5:3b",
                        help="Default: qwen2.5:3b. The model must be downloaded in Ollama first.",
                        key="ollama_model_name",
                    ).strip() or "qwen2.5:3b"
                    st.markdown(
                        "Install Ollama, then run this command once in PowerShell:\n\n"
                        "`ollama pull qwen2.5:3b`\n\n"
                        "Keep the Ollama app running while using MyMaizeCare."
                    )
                ollama_model = st.session_state.get("ollama_model_name", "qwen2.5:3b")
                if st.button("Generate Local AI Report", type="primary", use_container_width=True):
                    try:
                        method_name = (
                            xai_result["method"]
                            if xai_result and xai_result.get("image_hash") == image_hash
                            else None
                        )
                        with st.spinner(f"Generating report with local model {ollama_model}..."):
                            report = create_local_llm_report(
                                predicted_name,
                                confidence,
                                probabilities,
                                method_name,
                                ollama_model,
                            )
                        st.session_state["llm_report"] = {
                            "image_hash": image_hash,
                            "report": report,
                            "provider": "Ollama local",
                        }
                    except Exception as exc:
                        st.error(f"Local AI report could not be generated: {exc}")

                llm_result = st.session_state.get("llm_report")
                if llm_result and llm_result.get("image_hash") == image_hash:
                    st.markdown(llm_result["report"])
                    st.caption("Generated locally with Ollama • The uploaded image is not sent to a cloud LLM.")
    else:
        st.info("Upload or capture a maize leaf image to begin.")

    st.caption("MyMaizeCare • Diagnosis")
