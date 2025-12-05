import gradio as gr
import torch
import os
import sys
from PIL import Image
import uuid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hpsv3.inference import HPSv3RewardInferencer
try:
    import ImageReward as RM
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
except:
    RM = None
    create_model_and_transforms = None
    get_tokenizer = None
    print("ImageReward or HPSv2 dependencies not found. Skipping those models.")

from transformers import AutoProcessor, AutoModel

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if DEVICE == 'cuda' else torch.float32

# --- Model Configuration ---
MODEL_CONFIGS = {
    "HPSv3_7B": {
        "name": "HPSv3 7B",
        "type": "hpsv3"
    },
    "HPSv2": {
        "name": "HPSv2",
        "checkpoint_path": "your_path_to_HPS_v2_compressed.pt",
        "type": "hpsv2"
    },
    "ImageReward": {
        "name": "ImageReward v1.0",
        "checkpoint_path": "ImageReward-v1.0",
        "type": "imagereward"
    },
    "PickScore": {
        "name": "PickScore",
        "checkpoint_path": "your_path_to_pickscore",
        "type": "pickscore"
    },
    "CLIP": {
        "name": "CLIP ViT-H-14",
        "checkpoint_path": "/preflab/models/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "type": "clip"
    }
}

# --- Global Model Storage ---
current_models = {}
current_model_name = None

# --- Dynamic Model Loading Functions ---
def load_model(model_key, update_status_fn=None):
    """Load the specified model based on the model key."""
    global current_models, current_model_name
    
    if model_key == current_model_name and model_key in current_models:
        return current_models[model_key]
    
    if update_status_fn:
        update_status_fn(f"🔄 Loading {MODEL_CONFIGS[model_key]['name']}...")
    
    # Clear previous models to save memory
    current_models.clear()
    torch.cuda.empty_cache()
    
    config = MODEL_CONFIGS[model_key]
    
    try:
        if config["type"] == "hpsv3":
            model = HPSv3RewardInferencer(
                device=DEVICE, 
            )
        elif config["type"] == "hpsv2":
            model_obj, preprocess_train, preprocess_val = create_model_and_transforms(
                'ViT-H-14',
                'laion2B-s32B-b79K',
                precision='amp',
                device=DEVICE,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False
            )
            checkpoint = torch.load(config["checkpoint_path"], map_location=DEVICE, weights_only=False)
            model_obj.load_state_dict(checkpoint['state_dict'])
            model_obj = model_obj.to(DEVICE).eval()
            tokenizer = get_tokenizer('ViT-H-14')
            model = {"model": model_obj, "preprocess_val": preprocess_val, "tokenizer": tokenizer}
        elif config["type"] == "imagereward":
            model = RM.load(config["checkpoint_path"])
        elif config["type"] == "pickscore":
            processor = AutoProcessor.from_pretrained('/preflab/models/CLIP-ViT-H-14-laion2B-s32B-b79K')
            model_obj = AutoModel.from_pretrained(config["checkpoint_path"]).eval().to(DEVICE)
            model = {"model": model_obj, "processor": processor}
        elif config["type"] == "clip":
            model_obj = AutoModel.from_pretrained(config["checkpoint_path"]).to(DEVICE)
            processor = AutoProcessor.from_pretrained(config["checkpoint_path"])
            model = {"model": model_obj, "processor": processor}
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
        
        current_models[model_key] = model
        current_model_name = model_key
        
        if update_status_fn:
            update_status_fn(f"✅ {MODEL_CONFIGS[model_key]['name']} loaded successfully!")
        
        return model
    except Exception as e:
        error_msg = f"Error loading model {model_key}: {e}"
        print(error_msg)
        if update_status_fn:
            update_status_fn(f"❌ {error_msg}")
        return None

def score_with_model(model_key, image_paths, prompts):
    """Score images using the specified model."""
    model = load_model(model_key)
    if model is None:
        raise ValueError(f"Failed to load model {model_key}")
    
    config = MODEL_CONFIGS[model_key]
    
    if config["type"] == "hpsv3":
        rewards = model.reward(prompts, image_paths)
        return [reward[0].item() for reward in rewards]  # HPSv3 returns tensor with multiple values, take first
    elif config["type"] == "hpsv2":
        return score_hpsv2_batch(model, image_paths, prompts)
    elif config["type"] == "imagereward":
        return [model.score(prompt, image_path) for prompt, image_path in zip(prompts, image_paths)]
    elif config["type"] == "pickscore":
        return score_pickscore_batch(prompts, image_paths, model["model"], model["processor"])
    elif config["type"] == "clip":
        return score_clip_batch(model["model"], model["processor"], image_paths, prompts)
    else:
        raise ValueError(f"Unknown model type: {config['type']}")

def score_hpsv2_batch(model_dict, image_paths, prompts):
    """Score using HPSv2 model."""
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']
    tokenizer = model_dict['tokenizer']

    # 批量处理图片
    images = [preprocess_val(Image.open(p)).unsqueeze(0)[:,:3,:,:] for p in image_paths]
    images = torch.cat(images, dim=0).to(device=DEVICE)
    texts = tokenizer(prompts).to(device=DEVICE)
    with torch.no_grad():
        outputs = model(images, texts)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits_per_image = image_features @ text_features.T
        hps_scores = torch.diagonal(logits_per_image).cpu()
    return [score.item() for score in hps_scores]

def score_pickscore_batch(prompts, image_paths, model, processor):
    """Score using PickScore model."""
    pil_images = [Image.open(p) for p in image_paths]
    image_inputs = processor(
        images=pil_images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(DEVICE)
    
    text_inputs = processor(
        text=prompts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)
        return [scores[i, i].cpu().item() for i in range(len(prompts))]

def score_clip_batch(model, processor, image_paths, prompts):
    """Score using CLIP model."""
    pil_images = [Image.open(p) for p in image_paths]
    image_inputs = processor(
        images=pil_images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(DEVICE)
    
    text_inputs = processor(
        text=prompts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        scores = image_embs @ text_embs.T
        return [scores[i, i].cpu().item() for i in range(len(prompts))]

# Load default model
print("Loading default HPSv3 model...")
load_model("HPSv3_7B")
print("Model loaded successfully.")

# --- Helper Functions ---
def get_score_interpretation(score):
    """Returns a color-coded qualitative interpretation of the score."""
    if score is None:
        return ""
    
    if score < 0:
        color = "#ef4444"  # Modern red
        bg_color = "rgba(239, 68, 68, 0.1)"
        icon = "❌"
        feedback = "Poor Quality"
        comment = "The image has significant quality issues or doesn't match the prompt well."
    elif score < 5:
        color = "#f59e0b"  # Modern amber
        bg_color = "rgba(245, 158, 11, 0.1)"
        icon = "⚠️"
        feedback = "Needs Improvement"
        comment = "The image is acceptable but could be enhanced in quality or prompt alignment."
    elif score < 10:
        color = "#10b981"  # Modern emerald
        bg_color = "rgba(16, 185, 129, 0.1)"
        icon = "✅"
        feedback = "Good Quality"
        comment = "A well-crafted image that aligns nicely with the given prompt."
    else:  # score >= 10
        color = "#06d6a0"  # Vibrant teal
        bg_color = "rgba(6, 214, 160, 0.1)"
        icon = "⭐"
        feedback = "Excellent!"
        comment = "Outstanding quality and perfect alignment with the prompt."
    
    return f"""
    <div style='
        background: {bg_color};
        border: 2px solid {color};
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    '>
        <div style='font-size: 2rem; margin-bottom: 8px;'>{icon}</div>
        <h3 style='color: {color}; font-size: 1.4rem; font-weight: 700; margin: 8px 0;'>{feedback}</h3>
        <p style='color: #666; font-size: 0.95rem; margin: 0; line-height: 1.4;'>{comment}</p>
    </div>
    """

# --- Model Change Handler ---
def handle_model_change(model_key):
    """Handle model selection change."""
    global current_model_name
    
    if model_key != current_model_name:
        # Show loading status
        yield f"🔄 Loading {MODEL_CONFIGS[model_key]['name']}..."
        
        # Load the new model
        model = load_model(model_key)
        
        if model is not None:
            yield f"✅ Current model: {MODEL_CONFIGS[model_key]['name']}"
        else:
            yield f"❌ Failed to load {MODEL_CONFIGS[model_key]['name']}"
    else:
        yield f"✅ Current model: {MODEL_CONFIGS[model_key]['name']}"

# --- Prediction Function ---
def predict_score(image, prompt, model_name):
    """Takes Gradio inputs and returns the score, interpretation, and status."""
    if image is None:
        return None, "", "❌ Error: Please upload an image."
    if not prompt or not prompt.strip():
        return None, "", "❌ Error: Please enter a prompt."

    temp_dir = "temp_images_for_gradio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.png")
    
    try:
        Image.fromarray(image).save(temp_path)
        scores = score_with_model(model_name, [temp_path], [prompt])
        score = round(scores[0], 4)
        interpretation = get_score_interpretation(score)
        return score, interpretation, "✅ Analysis completed successfully!"
    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return None, "", f"❌ Processing error: {e}"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- Image Comparison Function ---
def compare_images(image1, image2, prompt, model_name):
    """Compare two images and determine which one is better based on the prompt."""
    if image1 is None or image2 is None:
        return None, None, "", "❌ Error: Please upload both images."
    if not prompt or not prompt.strip():
        return None, None, "", "❌ Error: Please enter a prompt."

    temp_dir = "temp_images_for_gradio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path1 = os.path.join(temp_dir, f"{uuid.uuid4()}_img1.png")
    temp_path2 = os.path.join(temp_dir, f"{uuid.uuid4()}_img2.png")
    
    try:
        Image.fromarray(image1).save(temp_path1)
        Image.fromarray(image2).save(temp_path2)
        
        # Get scores for both images
        scores = score_with_model(model_name, [temp_path1, temp_path2], [prompt, prompt])
        score1 = round(scores[0], 4)
        score2 = round(scores[1], 4)
        
        # Determine winner
        if score1 > score2:
            winner_text = f"🏆 **Image 1 is better!**\n\nImage 1 Score: **{score1}**\nImage 2 Score: **{score2}**\n\nDifference: **+{round(score1-score2, 4)}**"
        elif score2 > score1:
            winner_text = f"🏆 **Image 2 is better!**\n\nImage 1 Score: **{score1}**\nImage 2 Score: **{score2}**\n\nDifference: **+{round(score2-score1, 4)}**"
        else:
            winner_text = f"🤝 **It's a tie!**\n\nBoth images scored: **{score1}**"
        
        return score1, score2, winner_text, "✅ Comparison completed successfully!"
        
    except Exception as e:
        print(f"An error occurred during comparison: {e}")
        return None, None, "", f"❌ Processing error: {e}"
    finally:
        if os.path.exists(temp_path1):
            os.remove(temp_path1)
        if os.path.exists(temp_path2):
            os.remove(temp_path2)

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="HPSv3 - Human Preference Score v3") as demo:
    gr.HTML(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>🎨 HPSv3: Human Preference Score v3</h1>
        <p>Evaluate image quality and alignment with prompts with multiple models.</p>
        <p><a href="https://mizzenai.github.io/HPSv3.project/" target="_blank">🌐 Project Website</a> | 
            <a href="https://huggingface.co/papers/2508.03789" target="_blank">📄 Paper</a> | 
            <a href="https://github.com/MizzenAI/HPSv3" target="_blank">💻 Code</a></p>
    </div>
    """)
    
    # Global model selector
    with gr.Row():
        model_selector = gr.Dropdown(
            choices=[(config["name"], key) for key, config in MODEL_CONFIGS.items()],
            value="HPSv3_7B",
            label="🤖 Select Model",
        )
        model_status = gr.Textbox(
            label="Model Status",
            value=f"✅ Current model: {MODEL_CONFIGS['HPSv3_7B']['name']}",
            interactive=False,
            scale=2
        )
    
    with gr.Tabs():
        # Tab 1: Single Image Scoring
        with gr.TabItem("📊 Image Scoring"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("### 🖼️ **Upload & Describe**")
                        image_input = gr.Image(
                            type="numpy", 
                            label="Upload Image", 
                            height=450
                        )
                        prompt_input = gr.Textbox(
                            label="Prompt Description", 
                            placeholder="Describe what the image should represent...",
                            lines=3,
                            max_lines=5
                        )
                
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 🎯 **Quality Assessment**")
                        score_output = gr.Number(
                            label="Score", 
                            elem_id="score-output",
                            precision=4
                        )
                        interpretation_output = gr.Markdown(label="")
                        status_output = gr.Textbox(
                            label="Status", 
                            interactive=False
                        )
            submit_button = gr.Button(
                "🚀 Run Evaluation", 
                variant="primary",
                size="lg"
            )
            
            submit_button.click(
                fn=predict_score,
                inputs=[image_input, prompt_input, model_selector],
                outputs=[score_output, interpretation_output, status_output]
            )

            with gr.Group():
                gr.Examples(
                    examples=[
                        ["assets/example1.png", "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker, high resolution, vibrant colors"],
                        ["assets/example2.png", "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker, high resolution, vibrant colors"],
                    ],
                    inputs=[image_input, prompt_input],
                    outputs=[score_output, interpretation_output, status_output],
                    fn=lambda img, prompt: predict_score(img, prompt, "HPSv3_7B"),
                    cache_examples=False
                )
        
        # Tab 2: Image Comparison
        with gr.TabItem("⚖️ Image Comparison"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("### 🖼️ **Upload Images & Prompt**")
                        with gr.Row():
                            image1_input = gr.Image(
                                type="numpy", 
                                label="Image 1", 
                                height=300
                            )
                            image2_input = gr.Image(
                                type="numpy", 
                                label="Image 2", 
                                height=300
                            )
                        prompt_compare_input = gr.Textbox(
                            label="Prompt Description", 
                            placeholder="Describe what the images should represent...",
                            lines=3,
                            max_lines=5
                        )
                
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 🎯 **Comparison Results**")
                        score1_output = gr.Number(
                            label="Image 1 Score", 
                            precision=4
                        )
                        score2_output = gr.Number(
                            label="Image 2 Score", 
                            precision=4
                        )
                        comparison_result = gr.Markdown(label="Winner")
                        status_compare_output = gr.Textbox(
                            label="Status", 
                            interactive=False
                        )
            
            compare_button = gr.Button(
                "⚖️ Compare Images", 
                variant="primary",
                size="lg"
            )
            
            compare_button.click(
                fn=compare_images,
                inputs=[image1_input, image2_input, prompt_compare_input, model_selector],
                outputs=[score1_output, score2_output, comparison_result, status_compare_output]
            )

            with gr.Group():
                gr.Examples(
                    examples=[
                        ["assets/example1.png", "assets/example2.png", "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker, high resolution, vibrant colors"],
                        ["assets/example2.png", "assets/example1.png", "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker, high resolution, vibrant colors"],
                    ],
                    inputs=[image1_input, image2_input, prompt_compare_input],
                    outputs=[score1_output, score2_output, comparison_result, status_compare_output],
                    fn=lambda img1, img2, prompt: compare_images(img1, img2, prompt, "HPSv3_7B"),
                    cache_examples=False
                )

    # Model change handler
    model_selector.change(
        fn=handle_model_change,
        inputs=[model_selector],
        outputs=[model_status]
    )

def main():
    """Main function to launch the demo."""
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path=None,
        show_error=True,
    )

if __name__ == "__main__":
    main()
