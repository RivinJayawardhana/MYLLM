import torch
import tiktoken

from GPT_model import GPTModel, generate_text_simple
from Text_Preprocessing import create_dataloader_v1


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


# ---------------- Token Utilities ---------------- #

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


# ---------------- Loss Functions ---------------- #

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)

    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):

    total_loss = 0.

    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):

        if i >= num_batches:
            break

        loss = calc_loss_batch(
            input_batch,
            target_batch,
            model,
            device
        )

        total_loss += loss.item()

    return total_loss / num_batches


# ---------------- Evaluation ---------------- #

def evaluate_model(model, train_loader, val_loader, device, eval_iter):

    model.eval()

    with torch.no_grad():

        train_loss = calc_loss_loader(
            train_loader,
            model,
            device,
            eval_iter
        )

        val_loss = calc_loss_loader(
            val_loader,
            model,
            device,
            eval_iter
        )

    model.train()

    return train_loss, val_loss


# ---------------- Text Generation ---------------- #

def generate_and_print_sample(model, tokenizer, device, start_context):

    model.eval()

    context_size = model.pos_emb.weight.shape[0]

    encoded = text_to_token_ids(
        start_context,
        tokenizer
    ).to(device)

    with torch.no_grad():

        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )

    print(token_ids_to_text(token_ids, tokenizer))
    model.train()


# ---------------- Training ---------------- #

def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):

    train_losses = []
    val_losses = []
    track_tokens_seen = []

    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):

        model.train()

        for input_batch, target_batch in train_loader:

            optimizer.zero_grad()

            loss = calc_loss_batch(
                input_batch,
                target_batch,
                model,
                device,
            )

            loss.backward()

            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:

                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter,
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Epoch {epoch+1} "
                    f"Step {global_step} "
                    f"Train Loss={train_loss:.4f} "
                    f"Val Loss={val_loss:.4f}"
                )

        generate_and_print_sample(
            model,
            tokenizer,
            device,
            start_context,
        )

    return train_losses, val_losses, track_tokens_seen


# ---------------- Main ---------------- #

def main():

    file_path = "the-verse.txt"  # Path to your text file

    tokenizer = tiktoken.get_encoding("gpt2")

    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))

    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    torch.manual_seed(123)

    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=4e-4,
        weight_decay=0.1,
    )

    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )

    print("Training completed!")


if __name__ == "__main__":
    main()