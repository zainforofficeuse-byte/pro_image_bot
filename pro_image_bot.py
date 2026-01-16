import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance
# Rembg ko yahan se hata kar function k andar dal diya hai taake app fast load ho
import io
import numpy as np
import json
import requests
from bs4 import BeautifulSoup

# --- AI LIBRARIES (Lazy Load) ---
@st.cache_resource
def load_ai_model():
    """AI Model ko cache karta hai"""
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model
    except Exception as e:
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
    """Website se Title, Description aur Bullet Points nikalta hai"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return f"Error: Website not accessible (Status {response.status_code})"
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        features = []
        
        # 1. Meta Description
        meta_desc = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        if meta_desc:
            content = meta_desc.get('content', '').strip()
            if content: features.append(content[:100] + "...") # Limit length
            
        # 2. Bullet Points (ul/li usually contain specs)
        ul_tags = soup.find_all('ul')
        for ul in ul_tags:
            items = [li.get_text(strip=True) for li in ul.find_all('li') if len(li.get_text(strip=True)) > 5]
            if len(items) > 2 and len(items) < 15: # Filter junk lists
                features.extend(items[:5]) # Top 5 points utha lo
                break 
        
        # 3. Agar kuch na mile to Title utha lo
        if not features:
            title = soup.find('title')
            if title: features.append(title.get_text(strip=True))
            
        return ", ".join(features)
        
    except Exception as e:
        return f"Scraping Failed: {e}"

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro Image & Prompt Bot", layout="wide")

st.title("📸 Pro Image Studio & AI Prompts")

# --- TABS ---
tab1, tab2 = st.tabs(["🎨 Image Editor", "🧠 Smart AI Prompt Builder"])

# ==========================================
# TAB 1: IMAGE EDITOR
# ==========================================
with tab1:
    st.header("Basic Image Editor")
    
    add_shadow = st.sidebar.checkbox("Add Drop Shadow (Tab 1)", value=False)
    enhance_image = st.sidebar.checkbox("Enhance Quality (Tab 1)", value=True)
    product_scale = st.sidebar.slider("Product Size % (Tab 1)", 50, 100, 75, 5) / 100.0
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
        # Lazy Import: Taake app jaldi load ho
        from rembg import remove
        
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
        return final_canvas

    editor_files = st.file_uploader("Upload Images for Editing", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True, key="editor_uploader")
    
    if editor_files:
        if st.button("🚀 Process Images"):
            # Progress bar for better UX
            with st.spinner("Processing... (First time may take 1-2 mins to download AI models)"):
                for f in editor_files:
                    try:
                        res = process_single_image(f)
                        buf = io.BytesIO()
                        res.save(buf, format="JPEG", quality=95)
                        
                        col1, col2 = st.columns(2)
                        with col1: st.image(f, caption="Original", width=150)
                        with col2: st.image(res, caption="Pro Version", width=150)
                        
                        st.download_button(f"⬇️ Download {f.name}", data=buf.getvalue(), file_name=f"Pro_{f.name}.jpg", mime="image/jpeg")
                        st.divider()
                    except Exception as e:
                        st.error(f"Error processing {f.name}: {e}")

# ==========================================
# TAB 2: SMART AI PROMPT BUILDER
# ==========================================
with tab2:
    st.header("🧠 Smart Context-Aware Prompts")
    st.markdown("Product ka naam ya link dein, bot khud samjhega ke background kaisa hona chahiye.")
    st.divider()

    # --- STEP 1: INPUTS ---
    col1, col2 = st.columns([1, 1])
    with col1:
        product_name = st.text_input("Product Name (Required)", value="QCY AilyBuds Pro+")
        ref_link = st.text_input("Official Product Link (Optional)", placeholder="https://example.com/product")
        
        # --- NEW: AUTO SCRAPE BUTTON ---
        if st.button("🌐 Fetch Features from Link"):
            if ref_link:
                with st.spinner("Scraping website..."):
                    scraped_text = scrape_features_from_url(ref_link)
                    if "Error" not in scraped_text:
                        st.session_state['scraped_features'] = scraped_text
                        st.success("Features found!")
                    else:
                        st.error(scraped_text)
            else:
                st.warning("Pehle Link dalein!")

    with col2:
        prompt_file = st.file_uploader("Product Image (Optional - For Visual AI)", type=['png', 'jpg', 'jpeg', 'webp'], key="prompt_uploader")

    # Features Input (Auto-filled if scraped)
    default_val = st.session_state.get('scraped_features', "Adaptive ANC, Hi-Res Audio, 6-Mic AI Call, Bluetooth 5.3, Long Battery")
    features_input = st.text_area("Features List (Auto-filled from link or Type manually)", value=default_val, height=150)

    # Session State
    if 'style_options' not in st.session_state: st.session_state['style_options'] = {}
    if 'visual_context' not in st.session_state: st.session_state['visual_context'] = ""

    # --- STEP 2: ANALYZE & SUGGEST ---
    if st.button("🔍 Analyze Product & Suggest Styles"):
        with st.spinner("AI analyzing visuals and context..."):
            
            # A. Visual Analysis (BLIP)
            visual_desc = ""
            if prompt_file:
                try:
                    processor, model = load_ai_model()
                    if processor and model:
                        raw_image = Image.open(prompt_file).convert('RGB')
                        inputs = processor(raw_image, return_tensors="pt")
                        out = model.generate(**inputs)
                        visual_desc = processor.decode(out[0], skip_special_tokens=True)
                except: pass
            
            st.session_state['visual_context'] = visual_desc
            
            # B. Styles
            full_context_text = f"{product_name} {ref_link} {visual_desc}"
            st.session_state['style_options'] = get_background_suggestions(product_name, full_context_text)
            st.success("Analysis Complete! Please select a style below.")

    # --- STEP 3: SELECT & GENERATE ---
    if st.session_state['style_options']:
        st.divider()
        st.subheader("🎨 Choose a Style Direction")
        
        if st.session_state['visual_context']:
            st.caption(f"AI Visual Detection: *{st.session_state['visual_context']}*")

        style_choice = st.radio(
            "AI ne yeh 3 options suggest kiye hain:",
            list(st.session_state['style_options'].keys()),
            format_func=lambda x: f"**{x}** : {st.session_state['style_options'][x]}"
        )
        
        selected_bg_prompt = st.session_state['style_options'][style_choice]

        st.divider()
        if st.button("✨ Generate Final JSON Prompts"):
            
            lines = features_input.split(',')
            prompts_output = []
            LOCKED_LAYOUT = "wide horizontal banner"
            LOCKED_STYLE = "professional ecommerce product visualization"

            if st.session_state['visual_context']:
                base_comp = f"{st.session_state['visual_context']}, centered hero shot"
            else:
                base_comp = f"hero shot of {product_name}, centered"

            for line in lines:
                feat = line.strip()
                if not feat: continue
                
                vis_elem = "minimal elegant product focus"
                f_lower = feat.lower()
                if "anc" in f_lower or "noise" in f_lower: vis_elem = "soft sound wave ripples"
                elif "water" in f_lower or "proof" in f_lower: vis_elem = "fresh water splash or droplets"
                elif "battery" in f_lower or "power" in f_lower: vis_elem = "glowing energy ring"
                elif "bluetooth" in f_lower: vis_elem = "wireless signal waves"
                
                prompt_json = {
                    "product": product_name,
                    "layout": LOCKED_LAYOUT,
                    "background": selected_bg_prompt,
                    "composition": base_comp,
                    "visual_elements": vis_elem,
                    "text_overlay": {"headline": feat, "subline": "Premium Feature"},
                    "style": LOCKED_STYLE
                }
                prompts_output.append(prompt_json)

            st.subheader("✅ Final Output")
            st.json(prompts_output)
            json_str = json.dumps(prompts_output, indent=2)
            st.download_button("⬇️ Download JSON", data=json_str, file_name="smart_prompts.json", mime="application/json")
