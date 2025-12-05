import os
import json
import torch
import multiprocessing as mp
from tqdm import tqdm
from hpsv3.inference import HPSv3RewardInferencer
import argparse
from collections import defaultdict
import glob
import numpy as np
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
import ImageReward as RM
from transformers import AutoProcessor, AutoModel
def initialize_model_hpsv2(device, cp):
    model_dict = {}
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        'laion2B-s32B-b79K',
        precision='amp',
        device=device,
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

    checkpoint = torch.load(cp, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    tokenizer = get_tokenizer('ViT-H-14')

    model_dict['model'] = model
    model_dict['preprocess_val'] = preprocess_val
    return model_dict, tokenizer

def initialize_pickscore(device, checkpoint_path):
    processor = AutoProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    model = AutoModel.from_pretrained(checkpoint_path).eval().to(device)
    return model, processor

def initialize_aesthetic_model():
    import open_clip
    from os.path import expanduser
    from urllib.request import urlretrieve
    import torch.nn as nn

    def get_aesthetic_model(clip_model="vit_l_14"):
        """Load the aesthetic model with caching"""

        home = expanduser("~")
        cache_folder = home + "/.cache/emb_reader"
        path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
        if not os.path.exists(path_to_model):
            os.makedirs(cache_folder, exist_ok=True)
            url_model = (
                "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
            )
            urlretrieve(url_model, path_to_model)
        # Create appropriate linear layer
        if clip_model == "vit_l_14":
            m = nn.Linear(768, 1)
        elif clip_model == "vit_b_32":
            m = nn.Linear(512, 1)
        else:
            raise ValueError()
        m.load_state_dict(torch.load(path_to_model))
        m.eval()
        return m
    
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    amodel = get_aesthetic_model(clip_model="vit_l_14")
    return model, preprocess, amodel

def initialize_clip(device):
    """Initialize the CLIP model and processor."""
    model = AutoModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    return model.to(device), processor

def score_hpsv2_batch(model_dict, tokenizer, device, img_paths: list, prompts: list) -> list:
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']

    # 批量处理图片
    images = [preprocess_val(Image.open(p)).unsqueeze(0)[:,:3,:,:] for p in img_paths]
    images = torch.cat(images, dim=0).to(device=device)
    texts = tokenizer(prompts).to(device=device)
    with torch.no_grad():
        outputs = model(images, texts)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits_per_image = image_features @ text_features.T
        hps_scores = torch.diagonal(logits_per_image).cpu()
    return hps_scores

def score_pick_score_batch(prompts, images, model, processor, device):
    # preprocess
    pil_images = [Image.open(p) for p in images]
    image_inputs = processor(
        images=pil_images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)
        scores = torch.diagonal(scores).cpu()
    
    return scores


def score_aesthetic_batch(model, preprocess, aesthetic_model, device, img_paths: list) -> list:
    """Scores a batch of images using the aesthetic model."""
    images = [preprocess(Image.open(p)).unsqueeze(0) for p in img_paths]
    images = torch.cat(images, dim=0).to(device=device)
    with torch.no_grad():
        feat = model.encode_image(images)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        pred = aesthetic_model(feat).cpu()
    return pred

def score_clip_batch(model, processor, device, img_paths: list, prompts: list) -> list:
    """Scores a batch of images against prompts using CLIP."""
    # preprocess
    pil_images = [Image.open(p) for p in img_paths]
    image_inputs = processor(
        images=pil_images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        # score
        scores = image_embs @ text_embs.T
        scores = torch.diagonal(scores).cpu()
    
    return scores

def calculate_category_stats(data_dict):
    """Calculate statistics for each category"""
    stats = {}
    for category, data_list in data_dict.items():
        if not data_list:
            stats[category] = {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
            continue
            
        rewards = [item['reward'] for item in data_list]
        stats[category] = {
            'count': len(rewards),
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards))
        }
    total_mean = np.mean([stat['mean'] for stat in stats.values() if stat['count'] > 0])
    stats['OVERALL'] = {
        'count': sum(stat['count'] for stat in stats.values()),
        'mean': float(total_mean),
        'std': float(np.std([stat['mean'] for stat in stats.values() if stat['count'] > 0])),
        'min': float(min(stat['min'] for stat in stats.values() if stat['count'] > 0)),
        'max': float(max(stat['max'] for stat in stats.values() if stat['count'] > 0))
    }
    return stats

def print_stats(stats):
    print(f"{'Category':<30} {'Count':<8} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 78)
    for category, stat in stats.items():
        category_name = category  # Get folder name only
        print(f"{category_name:<30} {stat['count']:<8} {stat['mean']:<10.4f} {stat['std']:<10.4f} {stat['min']:<10.4f} {stat['max']:<10.4f}")
    
    # Calculate overall statistics
    if stats:
        all_counts = [stat['count'] for stat in stats.values()]
        all_means = [stat['mean'] for stat in stats.values() if stat['count'] > 0]
        if all_means:
            print("-" * 78)
            print(f"{'OVERALL':<30} {sum(all_counts):<8} {np.mean(all_means):<10.4f} {'':<10} {min([stat['min'] for stat in stats.values() if stat['count'] > 0]):<10.4f} {max([stat['max'] for stat in stats.values() if stat['count'] > 0]):<10.4f}")

def worker_process(process_id, process_dict, config_path, checkpoint_path, mode, device_id, dtype, batch_size, return_dict):
    """Worker process function that processes a chunk of data"""
    category_rewards = defaultdict(list)

    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    if mode == 'imagereward':
        model = RM.load("ImageReward-v1.0")
    elif mode == 'hpsv2':
        inferencer = initialize_model_hpsv2(device, checkpoint_path)
        model_dict, tokenizer = inferencer
    elif mode == 'hpsv3':
        inferencer = HPSv3RewardInferencer(config_path=config_path, checkpoint_path=checkpoint_path)
    elif mode == 'pickscore':
        model, processor = initialize_pickscore(device, checkpoint_path)
    elif mode == 'aesthetic':
        model, preprocess, aesthetic_model = initialize_aesthetic_model()
        model = model.to(device)
        aesthetic_model = aesthetic_model.to(device)
    elif mode == 'clip':
        model, processor = initialize_clip(device)
        model = model.to(device)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    for category, chunk_data in tqdm(process_dict.items(), total=len(process_dict), desc='Total', disable=not process_id == 0):
        processed_data = []
        # Process data in batches
        for batch_start in tqdm(range(0, len(chunk_data), batch_size), 
                                total=(len(chunk_data) + batch_size - 1) // batch_size, 
                                desc=f"Category {category}", disable=not process_id == 0):
            batch_end = min(batch_start + batch_size, len(chunk_data))
            image_paths = chunk_data[batch_start:batch_end]
            text_paths = [p[:-4]+'.txt' for p in image_paths]

            prompts = ['\n'.join(open(p, 'r').readlines()) for p in text_paths]

            with torch.no_grad(): 
                if mode == 'imagereward':
                    rewards = torch.tensor([model.score(prompt, image_path) for prompt, image_path in zip(prompts, image_paths)])
                elif mode == 'hpsv2':
                    rewards = score_hpsv2_batch(model_dict, tokenizer, device, image_paths, prompts)
                elif mode == 'hpsv3':
                    rewards = inferencer.reward(image_paths=image_paths, prompts=prompts)
                elif mode == 'pickscore':
                    rewards = score_pick_score_batch(prompts, image_paths, model, processor, device)
                elif mode == 'aesthetic':
                    rewards = score_aesthetic_batch(model, preprocess, aesthetic_model, device, image_paths)
                elif mode == 'clip':
                    rewards = score_clip_batch(model, processor, device, image_paths, prompts)
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
                
            torch.cuda.empty_cache()
            for i in range(len(image_paths)):
                if rewards.ndim == 2:
                    reward = rewards[i][0].item()
                else:
                    reward = rewards[i].item()
                processed_data.append({
                    'image_path': image_paths[i],
                    'reward': reward,
                    'prompt': prompts[i]
                })

        category_rewards[category] = processed_data

    return_dict[process_id] = {
        'data': category_rewards,
    }

def chunk_list(data_list, num_chunks):
    """Split list into roughly equal chunks"""
    chunk_size = len(data_list) // num_chunks
    remainder = len(data_list) % num_chunks
    
    chunks = []
    start = 0
    for i in range(num_chunks):
        # Add one extra item to first 'remainder' chunks
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end = start + current_chunk_size
        chunks.append(data_list[start:end])
        start = end
    
    return chunks

def main(config_path, checkpint_path, mode, image_folders, output_path, batch_size=16, num_processes=8, num_machines=1, machine_id=0):
    print(f"Config path: {config_path}")

    dtype = torch.bfloat16
    
    # Gather all data first
    folder_dict = {}
    for folder in image_folders:
        images = []
        for ext in ['.png', '.jpg']:
            images.extend(glob.glob(os.path.join(folder, "**", f"*{ext}"), recursive=True))
        machine_image_chunks = chunk_list(images, num_machines)
        image_list = machine_image_chunks[machine_id] if machine_id < len(machine_image_chunks) else []
        print(f"Folder {folder} total data points: {len(image_list)}")
        data_chunks = chunk_list(image_list, num_processes)
        print(f"Folder {folder} data split into {num_processes} chunks with sizes: {[len(chunk) for chunk in data_chunks]}")
        folder_dict[folder] = data_chunks

    per_process_folder_dict = []
    for i in range(num_processes):
        one_dict = {}
        for key, value in folder_dict.items():
            one_dict[key] = value[i] if i < len(value) else []
        per_process_folder_dict.append(one_dict)

    # Create manager for shared data between processes
    with mp.Manager() as manager:
        return_dict = manager.dict()
        processes = []
     
        # Start processes
        for i in range(num_processes):
            device_id = i % torch.cuda.device_count() if torch.cuda.is_available() else 0
            
            p = mp.Process(target=worker_process, 
                          args=(i, per_process_folder_dict[i], config_path, checkpint_path, mode, device_id, dtype, batch_size, return_dict))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        # Collect results from all processes
        all_processed_data = {}
        for i in range(num_processes):
            if i in return_dict:
                result = return_dict[i]
                process_data = result['data']
                # Merge data from each process
                for category, data_list in process_data.items():
                    if category not in all_processed_data:
                        all_processed_data[category] = []
                    all_processed_data[category].extend(data_list)
            else:
                print(f"No result from process {i}")
        
        # Calculate and print statistics for current machine
        if all_processed_data:
            stats = calculate_category_stats(all_processed_data)
            print(f"\n=== Machine {machine_id} Statistics ===")
            print_stats(stats)
    
    # Save results
    if num_machines > 1:
        # Save current machine's results
        machine_output_path = output_path.replace('.json', f'_machine_{machine_id}.json')
        with open(machine_output_path, "w") as f:
            json.dump(all_processed_data, f, indent=4)
        print(f"Machine {machine_id} results saved to {machine_output_path}")
        
        # If this is machine 0, try to gather results from all machines
        if machine_id == 0:
            print("Waiting for all machines to complete...")
            # Note: In practice, you might want to implement a proper synchronization mechanism
            # For now, this assumes all machine files exist
            final_result = {}
            for i in range(num_machines):
                machine_file = output_path.replace('.json', f'_machine_{i}.json')
                if os.path.exists(machine_file):
                    print(f"Loading results from machine {i}")
                    with open(machine_file, 'r') as f:
                        machine_data = json.load(f)
                    # Merge machine data
                    for category, data_list in machine_data.items():
                        if category not in final_result:
                            final_result[category] = []
                        final_result[category].extend(data_list)
                else:
                    print(f"Warning: Machine {i} results file not found: {machine_file}")
            
            # Calculate and print statistics for final results
            stats = calculate_category_stats(final_result)
            print("\n=== Final Combined Statistics ===")
            print_stats(stats)
            
            # Save final combined results with statistics
            final_output = {
                'statistics': stats,
                'data': final_result,
            }
            with open(output_path, "w") as f:
                json.dump(final_output, f, indent=4)
            print(f"Final combined results saved to {output_path}")
    else:
        # Single machine case - calculate statistics
        stats = calculate_category_stats(all_processed_data)
        print("\n=== Statistics ===")
        print_stats(stats)
        
        # Save results with statistics
        output_data = {
            'statistics': stats,
            'data': all_processed_data,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Results saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Process images with HPSv3 reward inference')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the model checkpoint file')
    parser.add_argument('--mode', type=str, choices=['imagereward','hpsv2', 'hpsv3', 'pickscore', 'aesthetic', 'clip'], default='hpsv3')
    parser.add_argument('--image_folders', type=str, nargs='+', required=True, help='List of image folder paths to process')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output JSON file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing (default: 16)')
    parser.add_argument('--num_processes', type=int, default=8, help='Number of processes to use (default: 8)')
    parser.add_argument('--num_machines', type=int, default=1, help='Total number of machines (default: 1)')
    parser.add_argument('--machine_id', type=int, default=0, help='ID of current machine (default: 0)')
    
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    args = parse_args()
    main(
        config_path=args.config_path,
        checkpint_path=args.checkpoint_path,
        mode=args.mode,
        image_folders=args.image_folders,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_processes=args.num_processes,
        num_machines=args.num_machines,
        machine_id=args.machine_id
    )
