
import os
import json
import torch
import multiprocessing as mp
from tqdm import tqdm
from hpsv3.inference import HPSv3RewardInferencer
from multiprocessing import Process, Queue
import math
import fire
import prettytable

def calc_rank_acc(score_sample, predict_sample):
    tol_cnt = 0.
    true_cnt = 0.
    for idx in range(len(score_sample)):
        item_base = score_sample[idx]["ranking"]
        item = predict_sample[idx]["rewards"]
        for i in range(len(item_base)):
            for j in range(i+1, len(item_base)):
                if item_base[i] > item_base[j]:
                    if item[i] >= item[j]:
                        tol_cnt += 1
                    elif item[i] < item[j]:
                        tol_cnt += 1
                        true_cnt += 1
                elif item_base[i] < item_base[j]:
                    if item[i] > item[j]:
                        tol_cnt += 1
                        true_cnt += 1
                    elif item[i] <= item[j]:
                        tol_cnt += 1
    return true_cnt / tol_cnt
                

def worker_process(process_id, data_chunk, config_path, checkpoint_path, batch_size, result_queue, mode):
    """
    Worker function for each process to handle a chunk of data
    """

    # Each process uses a different GPU (cycle through available GPUs)
    num_gpus = torch.cuda.device_count()
    device = f"cuda:{process_id % num_gpus}" if num_gpus > 0 else "cpu"
    dtype = torch.bfloat16
    
    print(f"Process {process_id} starting with device {device}, processing {len(data_chunk)} items")
    
    # Initialize model for this process
    inferencer = HPSv3RewardInferencer(config_path, checkpoint_path, device=device)

    process_correct = 0
    process_equal = 0
    process_results = []
    
    for batch_start in tqdm(range(0, len(data_chunk), batch_size), 
                            total=(len(data_chunk) + batch_size - 1) // batch_size, 
                            desc=f"Process {process_id}"):
        batch_end = min(batch_start + batch_size, len(data_chunk))
        batch_info = data_chunk[batch_start:batch_end]
        if mode == 'pair':
            image_paths_1 = [info["path1"] for info in batch_info]
            image_paths_2 = [info["path2"] for info in batch_info]
            prompts = [info["prompt"] for info in batch_info]

            with torch.no_grad(): 
                rewards_1 = inferencer.reward(image_paths=image_paths_1, prompts=prompts)
                rewards_2 = inferencer.reward(image_paths=image_paths_2, prompts=prompts)

            for i in range(len(batch_info)):
                info = batch_info[i]
                if rewards_1.ndim == 2:
                    reward_1, reward_2 = rewards_1[i][0].item(), rewards_2[i][0].item()
                else:
                    reward_1, reward_2 = rewards_1[i].item(), rewards_2[i].item()
                
                item_result = {
                    'reward_1': reward_1,
                    'reward_2': reward_2,
                    'correct': reward_1 > reward_2,
                    'equal': reward_1 == reward_2,
                    'info': info
                }
                process_results.append(item_result)
                
                print(f"Process {process_id} - Reward 1: {reward_1}, Reward 2: {reward_2}")
                if reward_1 > reward_2:
                    process_correct += 1
                if reward_1 == reward_2:
                    process_equal += 1

        elif mode == 'ranking':
            for item in batch_info:
                rewards =  inferencer.reward(image_paths=item["generations"], prompt=item["prompt"])
                predict_item = {
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "rewards": rewards
                }
                process_results.append(predict_item)
    # Put results in queue
    if mode == 'pair':
        result_queue.put({
            'process_id': process_id,
            'correct': process_correct,
            'equal': process_equal,
            'total': len(data_chunk),
            'results': process_results
        })
    elif mode == 'ranking':
        result_queue.put({
            'process_id': process_id,
            'results': process_results
        })

    print(f"Process {process_id} completed: {process_correct}/{len(data_chunk)} correct, {process_equal}/{len(data_chunk)} equal")

def main(test_json, config_path=None, batch_size=8, num_processes=8, checkpoint_path=None, mode='pair'):

    assert mode in ['pair', 'ranking'], "Mode must be either 'pair' or 'ranking'"
    assert checkpoint_path is not None, "Checkpoint path must be provided for inference"

    mp.set_start_method('spawn', force=True)

    info_list = json.load(open(test_json, "r"))

    print(f"Total items to process: {len(info_list)}")
    # Split data into chunks for each process
    chunk_size = math.ceil(len(info_list) / num_processes)
    data_chunks = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(info_list))
        if start_idx < len(info_list):
            chunk = info_list[start_idx:end_idx]
            data_chunks.append(chunk)
            print(f"Process {i}: {len(chunk)} items (indices {start_idx}-{end_idx-1})")
    
    # Ensure we have the right number of non-empty chunks
    actual_processes = len(data_chunks)
    print(f"Using {actual_processes} processes")
    
    # Create result queue and processes
    result_queue = Queue()
    processes = []
    
    print("Starting processes...")
    for i in range(actual_processes):
        p = Process(target=worker_process, args=(i, data_chunks[i], config_path, checkpoint_path, batch_size, result_queue, mode))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete and collect results
    all_results = []
    total_correct = 0
    total_equal = 0
    total_items = 0
    
    print("Waiting for processes to complete...")
    for i in range(actual_processes):
        result = result_queue.get()
        all_results.append(result)
        if mode == 'pair':
            total_correct += result['correct']
            total_equal += result['equal']
            total_items += result['total']

        print(f"Process {result['process_id']} finished: {result['correct']}/{result['total']} correct, {result['equal']}/{result['total']} equal")
    
    # Wait for all processes to join
    for p in processes:
        p.join()
    
    if mode == 'pair':
        aggregated_results = {
            'total_correct': total_correct,
            'total_equal': total_equal,
            'total_items': total_items,
            'accuracy': total_correct / total_items,
            'process_results': all_results
        }
        table = prettytable.PrettyTable()
        table.field_names = ["Total Items", "Correct", "Equal", "Incorrect", "Accuracy (%)"]

        incorrect = aggregated_results['total_items'] - aggregated_results['total_correct'] - aggregated_results['total_equal']
        accuracy_percent = 100 * aggregated_results['total_correct'] / aggregated_results['total_items']

        table.add_row([
            aggregated_results['total_items'],
            aggregated_results['total_correct'],
            aggregated_results['total_equal'],
            incorrect,
            f"{accuracy_percent:.2f}"
        ])
    elif mode == 'ranking':
        rank_acc = calc_rank_acc(info_list, all_results[0]['results'])
        table = prettytable.PrettyTable()
        table.field_names = ["Total Items", "Rank Accuracy (%)"]
        table.add_row([len(info_list), f"{rank_acc * 100:.2f}"])

    print(table)
    
if __name__ == "__main__":
    fire.Fire(main)
