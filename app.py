import streamlit as st
from diffusers import StableDiffusionPipeline
import torch, io, gc
from PIL import Image

#PAGE & GLOBAL STYLE
st.set_page_config(page_title="Chikankari AI Design Generator", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Poppins:wght@300;400;500&display=swap');

    body, .stApp {background:#f0f4f9;color:#333;font-family:'Poppins',sans-serif;}
    h1{font-family:'Playfair Display',serif !important;font-size:2.3rem !important;
       margin-bottom:.25rem;color:#3d3d3d;}

    /* Card */
    .card{background:#fff;border-radius:14px;padding:1.1rem 2rem 1.4rem;
          box-shadow:0 6px 14px rgba(0,0,0,.08);margin-bottom:1.3rem;}

    /* Input */
    .stTextInput input{background:#fafafa;border:1px solid #ccd4e0;border-radius:10px;
                       padding:.6em .9em;font-size:1rem;color:#222;}
    input::placeholder{color:rgba(0,0,0,.35)!important;font-style:italic;}

    /* Buttons */
    .stButton>button{background:linear-gradient(120deg,#7b68ee,#9d8bff);color:#fff;border:none;
                     border-radius:10px;padding:.6em 1.3em;font-weight:500;font-size:1rem;
                     transition:transform .1s ease;}
    .stButton>button:hover{transform:translateY(-2px);}
    .stDownloadButton>button{background:#e7eaf5;color:#333;border-radius:8px;font-size:.88rem;
                             padding:.45em .9em;}

    /* Slider label colour */
    .stSlider > div{color:#444;}

    .prompt-examples{font-size:.9rem;color:#5f5f5f;text-align:center;margin:-.35rem 0 1.05rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# HEADER
st.markdown("<h1 style='text-align:center'>Chikankari AI Design Generator</h1>",
            unsafe_allow_html=True)
st.markdown("<div style='text-align:center;font-size:1.05rem;'>Turn words into elegant embroidered patterns</div>",
            unsafe_allow_html=True)
st.markdown(
    "<div class='prompt-examples'>Try: <code>floral vines</code>, "
    "<code>geometric paisley border</code>, <code>lotus motif</code></div>",
    unsafe_allow_html=True,
)

# PROMPT HELPER 
SUFFIX = ", embroidery"

def build_prompt(text: str) -> str:
    txt = text.strip()
    return txt if any(k in txt.lower() for k in ("chikan", "embroid")) else txt + SUFFIX

def clear_vram():
    torch.cuda.empty_cache(); gc.collect()

# SESSION 
if "imgs" not in st.session_state:
    st.session_state.imgs = []

# INPUT CARD
_, mid, _ = st.columns([1, 2, 1])
with mid:
    

    user_txt = st.text_input("", placeholder="floral paisley embroidery")
    num      = st.slider("Number of designs", 1, 4, 2)
    generate = st.button("Generate design")

    

GENERATE DESIGNS  
if generate and user_txt.strip():
    st.session_state.imgs = []                      
    prompt = build_prompt(user_txt)

    try:
        with st.spinner("Stitching your designs?"):
            MODEL_ID, LORA = "runwayml/stable-diffusion-v1-5", "./outputs/chikankari_lora"
            gpu_ok = torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 5e9
            device, dtype = ("cuda", torch.float16) if gpu_ok else ("cpu", torch.float32)

            pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
            pipe.unet.load_attn_procs(LORA); pipe.to(device)

            for _ in range(num):
                im = pipe(prompt, num_images_per_prompt=1,
                          num_inference_steps=45, guidance_scale=7.5).images[0]
                st.session_state.imgs.append(im)
                if device == "cuda": clear_vram()

            pipe.to("cpu"); del pipe; clear_vram()

    except torch.cuda.OutOfMemoryError:
        clear_vram(); st.error("GPU ran out of memory ? try fewer designs or restart.")
    except Exception as e:
        clear_vram(); st.error(f"Error: {e}")


if st.session_state.imgs:
    
    st.markdown("### Your designs")

    cols = st.columns(len(st.session_state.imgs))
    for i, (c, im) in enumerate(zip(cols, st.session_state.imgs)):
        with c:
            st.image(im, use_container_width=True, caption=f"Design {i+1}")
            buf = io.BytesIO(); im.save(buf, format="PNG")
            st.download_button("Download", buf.getvalue(),
                               file_name=f"chikankari_design_{i+1}.png",
                               mime="image/png", key=f"dl{i}")
    
