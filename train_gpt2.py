import os
import csv
import random
import pickle
import argparse

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel, get_scheduler


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
class AMPDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


def load_sequences(data_path):
    """Read one sequence per row, uppercase, strip whitespace, drop empties,
    and de-duplicate."""
    with open(data_path, "r") as f:
        reader = csv.reader(f)
        raw = [row[0].strip().upper() for row in reader if row and row[0].strip()]

    before = len(raw)
    raw = list(dict.fromkeys(raw))  # dedup, preserves order
    after = len(raw)
    if before != after:
        print(f"Removed {before - after} duplicate sequences ({before} -> {after}).")
    return raw


def collate_fn(batch, pad_token_id):
    """Pads to the longest sequence in the batch and returns an attention mask."""
    max_len = max(len(w) for w in batch)
    input_ids, attention_mask = [], []
    for seq in batch:
        pad_len = max_len - len(seq)
        input_ids.append(list(seq) + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(seq) + [0] * pad_len)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
    )


def calculate_loss_and_accuracy(outputs, labels, attention_mask, pad_token_id, device):
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)
    shift_mask = attention_mask[..., 1:].contiguous().to(device)

    loss_fct = CrossEntropyLoss(ignore_index=pad_token_id, reduction="sum")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    not_ignore = shift_mask.bool() & shift_labels.ne(pad_token_id)
    num_targets = not_ignore.long().sum().item()
    if num_targets == 0:
        z = torch.tensor(0.0, device=device)
        return z, z

    _, preds = shift_logits.max(dim=-1)
    correct = (shift_labels == preds) & not_ignore
    accuracy = correct.float().sum() / num_targets
    loss = loss / num_targets
    return loss, accuracy


def run_epoch(model, dataloader, tokenizer, device, optimizer=None, scheduler=None):
    """Shared loop for train and eval."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    with torch.set_grad_enabled(is_train):
        for input_ids, attention_mask in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss, acc = calculate_loss_and_accuracy(
                outputs, input_ids, attention_mask, tokenizer.pad_token_id, device
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            total_acc += acc.item()
            n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
@torch.no_grad()
def generate_sequences(
    model,
    tokenizer,
    device,
    training_sequences,
    num_to_generate=1000,
    max_length=50,
    temperature=1.0,
    top_k=15,  # FIX 1: Reduced from 50 to fit small amino acid vocabulary
    top_p=0.95,
    min_len=5,
):
    """Generates candidate peptides and returns them with a length-normalized
    confidence score."""
    model.eval()
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    train_set = set(training_sequences)

    results = []
    seen = set()
    attempts = 0
    max_attempts = num_to_generate * 10

    while len(results) < num_to_generate and attempts < max_attempts:
        attempts += 1
        input_ids = torch.tensor([[tokenizer.cls_token_id]], device=device)

        generated_ids = []
        token_logprobs = []
        cur = input_ids
        for _ in range(max_length):
            out = model(cur)
            next_token_logits = out.logits[0, -1, :] / max(temperature, 1e-5)

            # top-k (FIX 2: Dynamic clamping to ensure k never exceeds vocabulary length)
            if top_k > 0:
                effective_k = min(top_k, next_token_logits.size(-1))
                topk_vals, topk_idx = torch.topk(next_token_logits, effective_k)
                mask = torch.full_like(next_token_logits, float("-inf"))
                mask[topk_idx] = topk_vals
                next_token_logits = mask

            # nucleus (top-p)
            probs = torch.softmax(next_token_logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cum_probs > top_p
            cutoff[1:] = cutoff[:-1].clone()
            cutoff[0] = False
            sorted_probs[cutoff] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()

            next_token = sorted_idx[torch.multinomial(sorted_probs, 1)]
            token_logprob = torch.log(probs[next_token] + 1e-12).item()

            if next_token.item() == tokenizer.sep_token_id:
                break

            generated_ids.append(next_token.item())
            token_logprobs.append(token_logprob)
            cur = torch.cat([cur, next_token.view(1, 1)], dim=1)

        if len(generated_ids) < min_len:
            continue

        seq = tokenizer.decode(generated_ids, skip_special_tokens=True).replace(" ", "").upper()
        if not seq or not set(seq) <= valid_aa:
            continue
        if seq in seen:
            continue
        seen.add(seq)

        mean_logprob = float(np.mean(token_logprobs))
        # Geometric mean of per-token probabilities: exp(mean(log p_i)).
        # Bounded in (0, 1], with 1.0 meaning the model was maximally
        # confident about every token it sampled. Equivalent to 1/perplexity.
        confidence = float(np.exp(mean_logprob))
        is_exact_match = seq in train_set
        results.append(
            {"sequence": seq, "confidence": confidence, "exact_match_to_training": is_exact_match}
        )

    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./train_plant_amps.csv")
    parser.add_argument("--vocab_path", default="./vocab.txt")
    parser.add_argument("--output_dir", default="./amp_model_checkpoint")
    parser.add_argument("--pkl_output_path", default="./amp_gpt2_model.pkl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience in epochs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # FIX 3: Set do_lower_case=True to match lowercase letters in vocab.txt
    tokenizer = BertTokenizer(
        vocab_file=args.vocab_path,
        do_lower_case=True,
        tokenize_chinese_chars=False,
    )

    raw_data = load_sequences(args.data_path)
    if len(raw_data) < 50:
        print(f"Warning: only {len(raw_data)} training sequences. A GPT-2 model this size "
              f"is very likely to memorize rather than generalize on a dataset this small; "
              f"treat generated candidates accordingly.")

    data_list = [tokenizer.encode(seq) for seq in raw_data]

    # Train/val split
    random.shuffle(data_list)
    n_val = max(1, int(len(data_list) * args.val_split))
    val_data, train_data = data_list[:n_val], data_list[n_val:]
    print(f"Train sequences: {len(train_data)} | Val sequences: {len(val_data)}")

    train_loader = DataLoader(
        AMPDataset(train_data), batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        AMPDataset(val_data), batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    config = GPT2Config(
        vocab_size=len(tokenizer.vocab),
        n_positions=50,
        n_ctx=50,
        n_embd=256,
        n_layer=6,
        n_head=8,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
        resid_pdrop=0.15,
        embd_pdrop=0.15,
        attn_pdrop=0.15,
    )
    model = GPT2LMHeadModel(config=config).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=100,
        num_training_steps=args.epochs * len(train_loader),
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    print("Starting LLM training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, tokenizer, device, optimizer, scheduler)
        val_loss, val_acc = run_epoch(model, val_loader, tokenizer, device)

        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            with open(args.pkl_output_path, "wb") as f:
                pickle.dump({"config": config, "state_dict": model.state_dict()}, f)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: no val improvement for {args.patience} epochs.")
                break

    print(f"Best val_loss={best_val_loss:.4f}. Model saved to {args.output_dir} and {args.pkl_output_path}")

    best_model = GPT2LMHeadModel.from_pretrained(args.output_dir).to(device)
    candidates = generate_sequences(best_model, tokenizer, device, training_sequences=raw_data)

    exact = sum(c["exact_match_to_training"] for c in candidates)
    print(f"Generated {len(candidates)} candidates: {exact} exact matches to training data, "
          f"{len(candidates) - exact} novel.")

    with open(os.path.join(args.output_dir, "generated_candidates.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sequence", "Confidence", "ExactMatchToTraining"])
        for c in candidates:
            writer.writerow([c["sequence"], c["confidence"], c["exact_match_to_training"]])


if __name__ == "__main__":
    main()
