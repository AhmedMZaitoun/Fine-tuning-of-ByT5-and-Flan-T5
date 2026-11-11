import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import random
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, set_seed
from peft import PeftModel, PeftConfig

from tokenization.tokenization_utils import smi_tokenizer_spaces

device = "cuda" if torch.cuda.is_available() else "cpu"
template = "{sentence}"


def load_arguments():
    parser = argparse.ArgumentParser()

    # Base model and adapter paths
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the original pretrained base model")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to the saved PEFT adapter (output folder from training)")

    # Data
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path to the test CSV file (must have an 'Input' column)")
    parser.add_argument("--output_path", type=str, default="predictions.csv",
                        help="Where to save the predictions CSV")

    # Tokenisation
    parser.add_argument("--tokenization", type=str, default="none",
                        choices=['none', 'map', 'shrink', 'map_shrink', 'spaces', 'shrink_spaces'])

    # Generation settings
    parser.add_argument("--max_source_len", type=int, default=96)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--num_beams", type=int, default=5,
                        help="Beam search width (1 = greedy)")
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args


def main():
    args = load_arguments()
    print(args)

    # Reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    # ── Load tokeniser and base model, then attach the LoRA adapter ──
    print(f"\nLoading base model from: {args.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, legacy=False)
    base_model = T5ForConditionalGeneration.from_pretrained(args.base_model_path)

    print(f"Loading PEFT adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model = model.merge_and_unload()   # merge LoRA weights into base for faster inference
    model.eval()
    model.to(device)
    print(f"Model loaded on {device}\n")

    # ── Load test data ──
    df = pd.read_csv(args.test_path)
    if "Input" not in df.columns:
        raise ValueError("Test CSV must contain an 'Input' column.")

    inputs_raw = df["Input"].tolist()
    ground_truth = df["Output"].tolist() if "Output" in df.columns else None

    # ── Apply tokenisation pre-processing if needed ──
    def preprocess(smiles):
        if args.tokenization == "spaces":
            return smi_tokenizer_spaces(smiles)
        return smiles

    inputs_processed = [template.replace("{sentence}", preprocess(s)) for s in inputs_raw]

    # ── Run inference in batches ──
    all_predictions = []

    print(f"Running inference on {len(inputs_processed)} samples (batch size {args.batch_size}, beams {args.num_beams})...")
    for i in tqdm(range(0, len(inputs_processed), args.batch_size)):
        batch = inputs_processed[i: i + args.batch_size]

        encoded = tokenizer(
            batch,
            max_length=args.max_source_len,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=True)
        all_predictions.extend(decoded)

    # ── Save results ──
    results = pd.DataFrame({
        "Input": inputs_raw,
        "Prediction": all_predictions,
    })

    if ground_truth is not None:
        results["Ground_Truth"] = ground_truth
        exact_matches = sum(p == g for p, g in zip(all_predictions, ground_truth))
        em_score = 100 * exact_matches / len(ground_truth)
        print(f"\nExact Match Score: {em_score:.2f}% ({exact_matches}/{len(ground_truth)})")
        results["Correct"] = [p == g for p, g in zip(all_predictions, ground_truth)]

    results.to_csv(args.output_path, index=False)
    print(f"\nPredictions saved to: {args.output_path}")


if __name__ == "__main__":
    main()
