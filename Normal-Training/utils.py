import numpy as np
import base64
import torch
import csv
import torch.nn.functional as F

def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Both strings must be of the same length")
    return sum(ch1 != ch2 for ch1, ch2 in zip(str1, str2))

def tensor_to_hash(y):
    """Convert the model's ouput to a Base64 encoded string (hash)."""
    # tensor = (tensor+1)/2*255.0
    y = y*255.0
    uint8_tensor = y.to(torch.uint8)
    uint8_array = y.cpu().detach().numpy().astype(np.uint8) ##hash bin with int
    byte_array = uint8_array.tobytes()
    encoded_str = base64.b64encode(byte_array).decode("utf-8") ##hashes
    binary_array = np.unpackbits(np.frombuffer(byte_array, dtype=np.uint8))
    binary_strings = ''.join(str(i) for i in binary_array) ##010101010

    return uint8_tensor, binary_strings, encoded_str

def calculate_acc(y_pred = None, y_true=None, hashes_csv = '/home/yuchen/code/verified_phash/Normal-Training/coco-val.csv'):
    hashes = []
    with open(hashes_csv) as f:
        r = csv.reader(f)
        for line in r:
            h = torch.tensor(np.array(list(base64.b64decode(line[1])), dtype=np.uint8)).float()/255.0
            # h = np.unpackbits(np.frombuffer(base64.b64decode(line[1]), dtype=np.uint8)) #into 01010110
            hashes.append(h)
    hashes_tensor = torch.stack(hashes).cuda()
    y_pred_expanded = y_pred.unsqueeze(1).expand(-1, hashes_tensor.size(0), -1)
    mse = torch.nn.functional.mse_loss(y_pred_expanded, hashes_tensor.unsqueeze(0), reduction='none').mean(2)
    min_mse_indices = torch.argmin(mse, dim=1)  # Shape [256]
    selected_hashes = hashes_tensor[min_mse_indices]
    correct_matches = torch.all(selected_hashes == y_true, dim=1)
    acc = torch.sum(correct_matches).item()/len(y_true)
    return acc


def evaluate(model, evasion_loss, val_dataloader, use_cuda):
    loss = 0
    hamming = 0
    acc = 0
    model.eval()
    with torch.no_grad():
        for data in val_dataloader:
            x, y = data
            x = x.type(torch.float32)
            y = y.type(torch.float32)
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
            y_pred = model(x)
            loss += evasion_loss(y_pred, y).item()
    model.train()
    return loss / len(val_dataloader)


def check_cuda():
    return torch.cuda.is_available() and torch.cuda.device_count() > 0

# def collision_loss_fn(pred_hashes, original_indices, margin=1.0):
#     # Compute the pairwise cosine similarity between hashes in the batch
#     similarity_matrix = torch.nn.functional.cosine_similarity(pred_hashes[:, None], pred_hashes[None, :], dim=-1)
#     # Zero out the diagonal (self-similarity) and pairs of augmented images from the same original
#     for i in range(len(original_indices)):
#         for j in range(i + 1, len(original_indices)):
#             if original_indices[i] == original_indices[j]:
#                 similarity_matrix[i, j] = 0
#                 similarity_matrix[j, i] = 0
#     # Apply the margin and compute the loss
#     loss = torch.max(similarity_matrix - margin, torch.zeros_like(similarity_matrix))
#     return loss.mean()
#
#
# import torch

def collision_loss_fn(y_p, margin=3000):
    """
    Compute the margin collision loss for a batch of predictions. The loss is designed to penalize
    predictions where the pairwise L1 distance for any individual feature is less than a specified margin.

    Parameters:
        y_p (torch.Tensor): A batch of predictions with shape (batch_size, feature_size).
        margin (int, optional): The margin threshold for each feature in the pairwise L1 distance. Defaults to 1800.

    Returns:
        torch.Tensor: The computed loss.
    """
    batch_size, feature_size = y_p.shape

    # Expand y_p to form all pairs (two different expansions for broadcasting)
    y_p1 = y_p.unsqueeze(1).expand(-1, batch_size, -1)  # Shape (batch_size, batch_size, feature_size)
    y_p2 = y_p.unsqueeze(0).expand(batch_size, -1, -1)  # Shape (batch_size, batch_size, feature_size)

    # Compute pairwise L1 distance using broadcasting for each feature
    pairwise_l1_distance = torch.abs(y_p1 - y_p2).sum(dim=2)  # Shape (batch_size, batch_size)
    # Apply the margin condition to each feature individually
    loss = F.relu(margin - pairwise_l1_distance)  # Apply margin threshold, shape remains (batch_size, batch_size, feature_size)

    # Mask out the diagonal elements since we don't want to calculate loss for them
    # mask = torch.eye(batch_size, dtype=torch.bool, device=y_p.device).unsqueeze(-1)
    mask = torch.eye(batch_size, dtype=torch.bool, device=y_p.device)
    loss = loss * (~mask)

    # Sum all elements and average across non-diagonal pairs and features
    total_loss = loss.sum() / (batch_size * (batch_size - 1))

    return total_loss

def deranged_permutation(n):
    """Generate a derangement of indices for tensor of length n."""
    while True:
        p = torch.randperm(n)
        if all(p != torch.arange(n)):
            return p

def pgd_linf(model, x, y, loss_fct, epsilon=0.251, alpha=0.01, num_iter=20, randomize=True):
    """ Construct  adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(x, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(x, requires_grad=True)
    for t in range(num_iter):
        y_p = model(x + delta)
        y_bad = y[deranged_permutation(len(y_p))]
        loss = loss_fct(y_p, y_bad)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()

def fgsm(model, x, y, loss_fct, epsilon=0.1, randomize=True):
    """ Construct FGSM adversarial examples on the examples X"""
    # model.eval()
    # # Ensure no gradients are updated in the model parameters
    # with torch.no_grad():
    #     original_state = {param: param.requires_grad for param in model.parameters()}
    #     for param in model.parameters():
    #         param.requires_grad = False

    if randomize:
        delta = torch.rand_like(x, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(x, requires_grad=True)
    y_p = model(x + delta)
    y_bad = y[deranged_permutation(len(y_p))]
    loss = loss_fct(y_p, y_bad)
    loss.backward()

    # # Restore original requires_grad states
    # for param in model.parameters():
    #     param.requires_grad = original_state[param]
    # model.train()

    return epsilon * delta.grad.detach().sign()
