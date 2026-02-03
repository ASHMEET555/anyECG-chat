import torch
import argparse
import json
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from anyecg.utils import Collate_Fn
from anyecg.ecg_language_modeling import ECG_Language_Model

# --- ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='Path to your test_custom.json')
parser.add_argument('--ecg_model_ckpt', type=str, required=True, help='Path to ecg_model.pth')
parser.add_argument('--projection_ckpt', type=str, required=True, help='Path to projection.pth')
parser.add_argument('--output_dir', type=str, default='./output/inference_results', help='Where to save results')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--temperature', type=float, default=0.6, help='Generation temperature')
args = parser.parse_args()

def run_inference():
    # 1. Setup Output Directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 2. Load Model
    print("Loading Model...")
    # Note: We initialize with defaults. The weights we load next will handle the rest.
    model = ECG_Language_Model()
    
    # Load the trained weights
    print(f"Loading ECG weights from: {args.ecg_model_ckpt}")
    model.ecg_model.load_state_dict(torch.load(args.ecg_model_ckpt))
    
    print(f"Loading Projection weights from: {args.projection_ckpt}")
    model.projection.load_state_dict(torch.load(args.projection_ckpt))
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 3. Load Data
    print(f"Loading Data from: {args.data_path}")
    dataset = load_dataset('json', data_files=args.data_path)['train']
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=Collate_Fn(), num_workers=4)

    # 4. Run Inference Loop
    results = []
    print("Starting Inference...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            ecgs, messages = batch
            
            # The model expects specific message formatting
            # messages is a list of lists. We need the "content" part for input.
            # Depending on how Collate_Fn works, 'messages' might be slightly different.
            # But usually for stage1, it passes standard conversation dicts.
            
            # Helper to generate
            # ecg_chat handles the tokenization and generation internally
            ecgs = [ecg.to(device) for ecg in ecgs]

            generated_texts = model.ecg_chat(ecgs, messages, args.temperature)
            
            # Save inputs and outputs for inspection
            for i in range(len(generated_texts)):
                # Extract ground truth from the input message structure
                # Typically messages[i][1] is the assistant response (Target)
                # But in testing, we only provide the Question.
                # Let's try to grab the answer from the original dataset row if possible, 
                # but here we rely on what the dataloader gives.
                
                ground_truth = "N/A"
                if len(messages[i]) > 1:
                    ground_truth = messages[i][1].get('content', '')

                results.append({
                    "question": messages[i][0].get('content', ''),
                    "ground_truth": ground_truth,
                    "generated_report": generated_texts[i]
                })

    # 5. Save Results
    output_file = os.path.join(args.output_dir, 'predictions.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Done! Results saved to {output_file}")

if __name__ == "__main__":
     run_inference()
#()CUDA_VISIBLE_DEVICES=1 python custom_inference.py \
#   --data_path ./data/test_custom.json \
#   --ecg_model_ckpt ./output/final_stage1_run/step_530/ecg_model.pth \
#   --projection_ckpt ./output/final_stage1_run/step_530/projection.pth \
#   --output_dir ./output/inference_results