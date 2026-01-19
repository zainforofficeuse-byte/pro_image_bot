import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance
import io
import numpy as np
import json
import requests
from bs4 import BeautifulSoup

# --- PAGE CONFIG (Sab se pehle ana chahiye) ---
st.set_page_config(page_title="Pro Image & Prompt Bot", layout="wide")

# --- AI LIBRARIES (Lazy Load with Status) ---
@st.cache_resource
def load_ai_model():
    """AI Model ko cache karta hai"""
    status = st.empty()
    status.info("⏳ AI Model Loading... (Sirf pehli baar waqt lagega)")
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        status.empty() # Remove message when done
        return processor, model
    except Exception as e:
        status.error(f"AI Model Error: {e}")
        return None, None

# --- HELPER: BACKGROUND SUGGESTIONS ---
def get_background_suggestions(query_text, visual_text):
    text = (query_text + " " + visual_text).lower()
    options = {}
    
    if any(x in text for x in ['earbud', 'headphone', 'audio', 'watch', 'phone', 'laptop', 'tech', 'anc']):
        options = {
            "Premium Studio": "soft off-white premium studio background with gentle shadows",
            "High-Tech Dark": "sleek dark background with subtle blue tech glow lines",
            "Lifestyle": "soft blurred modern living room interior background"
        }
    elif any(x in text for x in ['cream', 'oil', 'skin', 'beauty', 'serum', 'bottle']):
        options = {
            "Minimalist Water": "clean white background with soft water ripples",
            "Natural Organic": "soft beige background with botanical leaf shadows",
            "Luxury Gold": "premium silk texture background with warm lighting"
        }
    elif any(x in text for x in ['shoe', 'sneaker', 'wear', 'cloth', 'bag']):
        options = {
            "Urban Concrete": "concrete texture background with dramatic lighting",
            "Studio Clean": "infinite grey background with sharp shadows",
            "Vibrant Pop": "vibrant pastel colored background"
        }
    else:
        options = {
            "Clean Studio": "neutral soft studio background",
            "Gradient Modern": "soft cool-grey gradient background",
            "Warm Indoor": "warm ambient indoor lighting background"
        }
    return options

# --- HELPER: SCRAPE FEATURES ---
def scrape_features_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return f"Error: Status {response.status_code}"
        
        soup = BeautifulSoup(response.content, 'html.parser')
        features = []
        
        # Meta Desc
        meta_desc = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        if meta_desc:
            content = meta_desc.get('content', '').strip()
            if content: features.append(content[:100] + "...")
            
        # Ul/Li Bullet Points
        ul_tags = soup.find_all('ul')
        for ul in ul_tags:
            items = [li.get_text(strip=True) for li in ul.find_all('li') if len(li.get_text(strip=True)) > 5]
            if len(items) > 2 and len(items) < 15:
                features.extend(items[:5])
                break 
        
        if not features:
            title = soup.find('title')
            if title: features.append(title.get_text(strip=True))
            
        return ", ".join(features)
    except Exception as e:
        return f"Scraping Failed: {e}"

# --- MAIN UI ---
st.title("📸 Pro Image Studio & AI Prompts")

# --- TABS ---
tab1, tab2 = st.tabs(["🎨 Image Editor", "🧠 Smart AI Prompt Builder"])

# ==========================================
# TAB 1: IMAGE EDITOR
# ==========================================
with tab1:
    st.header("Basic Image Editor")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1: add_shadow = st.checkbox("Add Shadow", value=False)
    with col_s2: enhance_image = st.checkbox("Enhance Quality", value=True)
    with col_s3: product_scale = st.slider("Size %", 50, 100, 75, 5) / 100.0
    
    CANVAS_SIZE = (1080, 1080)
    BACKGROUND_COLOR = (255, 255, 255)

    def add_drop_shadow_effect(image, offset=(0, 20), shadow_color=(0, 0, 0, 100), blur_radius=20):
        try:
            mask = image.split()[3]
            shadow = Image.new("RGBA", image.size, shadow_color)
            shadow.putalpha(mask)
            shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))
            
            canvas_w = image.width + abs(offset[0]) + 50
            canvas_h = image.height + abs(offset[1]) + 50
            canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            
            shadow_pos = (25 + offset[0], 25 + offset[1])
            canvas.paste(shadow, shadow_pos, shadow)
            obj_pos = (25, 25)
            canvas.paste(image, obj_pos, image)
            return canvas.crop(canvas.getbbox())
        except: return image

    def process_single_image(uploaded_file):
        status_box = st.empty()
        status_box.info("⏳ Removing Background... (Checking Libraries)")
        
        try:
            # Lazy Import inside function to prevent app freeze
            from rembg import remove
            
            status_box.info("⏳ Processing Image...")
            original = Image.open(uploaded_file)
            no_bg = remove(original)
            
            if no_bg.getbbox(): no_bg = no_bg.crop(no_bg.getbbox())
            
            if enhance_image:
                enhancer = ImageEnhance.Sharpness(no_bg)
                no_bg = enhancer.enhance(1.2)
                enhancer = ImageEnhance.Color(no_bg)
                no_bg = enhancer.enhance(1.1)
            
            if add_shadow: final_obj = add_drop_shadow_effect(no_bg)
            else: final_obj = no_bg
                
            final_canvas = Image.new("RGB", CANVAS_SIZE, BACKGROUND_COLOR)
            target_w = int(CANVAS_SIZE[0] * product_scale)
            target_h = int(CANVAS_SIZE[1] * product_scale)
            final_obj.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
            
            bg_w, bg_h = final_canvas.size
            img_w, img_h = final_obj.size
            offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
            
            if final_obj.mode == 'RGBA': final_canvas.paste(final_obj, offset, final_obj)
            else: final_canvas.paste(final_obj, offset)
            
            status_box.empty()
            return final_canvas
            
        except ImportError:
            status_box.error("Error: 'rembg' library missing. Terminal mein `pip install rembg` likhein.")
            return None
        except Exception as e:
            status_box.error(f"Error: {e}")
            return None

    editor_files = st.file_uploader("Upload Images", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True, key="editor_uploader")
    
    if editor_files and st.button("🚀 Process Images"):
        for f in editor_files:
            res = process_single_image(f)
            if res:
                buf = io.BytesIO()
                res.save(buf, format="JPEG", quality=95)
                col1, col2 = st.columns(2)
                with col1: st.image(f, caption="Original", width=150)
                with col2: st.image(res, caption="Pro Version", width=150)
                st.download_button(f"⬇️ Download {f.name}", data=buf.getvalue(), file_name=f"Pro_{f.name}.jpg", mime="image/jpeg")
                st.divider()

# ==========================================
# TAB 2: SMART AI PROMPT BUILDER
# ==========================================
with tab2:
    st.header("🧠 Smart Context-Aware Prompts")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        product_name = st.text_input("Product Name", value="QCY AilyBuds Pro+")
        ref_link = st.text_input("Product Link", placeholder="https://example.com")
        
        if st.button("🌐 Fetch Features"):
            if ref_link:
                with st.spinner("Scraping..."):
                    scraped = scrape_features_from_url(ref_link)
                    st.session_state['scraped_features'] = scraped
                    if "Error" not in scraped: st.success("Done!")
                    else: st.error(scraped)

    with col2:
        prompt_file = st.file_uploader("Product Image (For AI)", type=['png', 'jpg', 'jpeg', 'webp'], key="prompt_uploader")

    default_val = st.session_state.get('scraped_features', "Adaptive ANC, Hi-Res Audio, 6-Mic AI Call, Bluetooth 5.3, Long Battery")
    features_input = st.text_area("Features", value=default_val, height=100)

    if 'style_options' not in st.session_state: st.session_state['style_options'] = {}
    if 'visual_context' not in st.session_state: st.session_state['visual_context'] = ""

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing... (Downloading AI Model if first time)"):
            
            # A. Visual Analysis
            visual_desc = ""
            if prompt_file:
                processor, model = load_ai_model() # This is cached
                if processor and model:
                    try:
                        raw_image = Image.open(prompt_file).convert('RGB')
                        inputs = processor(raw_image, return_tensors="pt")
                        out = model.generate(**inputs)
                        visual_desc = processor.decode(out[0], skip_special_tokens=True)
                    except: pass
            
            st.session_state['visual_context'] = visual_desc
            
            # B. Styles
            full_context = f"{product_name} {ref_link} {visual_desc}"
            st.session_state['style_options'] = get_background_suggestions(product_name, full_context)
            st.success("Complete!")

    if st.session_state['style_options']:
        st.divider()
        if st.session_state['visual_context']:
            st.caption(f"AI Saw: *{st.session_state['visual_context']}*")

        style_choice = st.radio("Choose Style:", list(st.session_state['style_options'].keys()))
        selected_bg = st.session_state['style_options'][style_choice]

        if st.button("✨ Generate JSON"):
            lines = features_input.split(',')
            prompts_output = []
            
            base_comp = f"{st.session_state['visual_context']}, centered hero shot" if st.session_state['visual_context'] else f"hero shot of {product_name}, centered"

            for line in lines:
                feat = line.strip()
                if not feat: continue
                
                vis_elem = "minimal product focus"
                f_lower = feat.lower()
                if "anc" in f_lower or "noise" in f_lower: vis_elem = "sound wave ripples"
                elif "water" in f_lower: vis_elem = "water splash"
                elif "battery" in f_lower: vis_elem = "glowing energy icon"
                
                prompt_json = {
                    "product": product_name,
                    "layout": "wide horizontal banner",
                    "background": selected_bg,
                    "composition": base_comp,
                    "visual_elements": vis_elem,
                    "text_overlay": {"headline": feat, "subline": "Premium Feature"},
                    "style": "professional ecommerce visualization"
                }
                prompts_output.append(prompt_json)

            st.json(prompts_output)
            st.download_button("⬇️ Download JSON", data=json.dumps(prompts_output, indent=2), file_name="prompts.json", mime="application/json")
