import torch
import numpy as np
def load_hash_matrix(path='nerualhash/neuralhash_128x96_seed1.dat'):
    """
    Loads the output hash matrix multiplied with the network logits.
    """
    seed1 = open(path, 'rb').read()[128:]
    seed1 = np.frombuffer(seed1, dtype=np.float32)
    seed1 = seed1.reshape([96, 128])
    return seed1
# Example usage
def compute_hash(logits, seed=None, binary=False, as_string=False):
    """
    Computes the final hash based on the network logits.
    """
    if seed is None:
        seed = load_hash_matrix()
    if type(seed) is torch.Tensor and type(logits) is torch.Tensor:
        seed = seed.to(logits.device)
        outputs = logits.squeeze().unsqueeze(1)
        hash_output = torch.mm(seed, outputs).flatten()
    else:
        if type(logits) is torch.Tensor:
            logits = logits.detach().cpu().numpy()
        if type(seed) is torch.Tensor:
            seed = seed.cpu().numpy()
        hash_output = seed.dot(logits.flatten())
    hash_bits = ''.join(['1' if it >= 0 else '0' for it in hash_output])
    hash_hex = '{:0{}x}'.format(int(hash_bits, 2), len(hash_bits) // 4)
    if binary:
        if as_string:
            return hash_bits
        hash_bits = torch.tensor([int(b) for b in hash_bits])
        hash_bits = hash_bits.to(logits.device)
        return hash_bits
    else:
        return hash_hex

def hash_to_binary_tensor(hash_string):
    # Convert the hash to a binary string
    bin_string = bin(int(hash_string, 16))[2:].zfill(len(hash_string) * 4)
    # Convert the binary string to a tensor of 0s and 1s
    return torch.tensor([int(bit) for bit in bin_string], dtype=torch.float32)

def hamming_distance(hash1, hash2):
    # Convert hashes to binary tensors
    tensor1 = hash_to_binary_tensor(hash1)
    tensor2 = hash_to_binary_tensor(hash2)
    # Calculate the Hamming distance
    distance = torch.norm(tensor1 - tensor2, p=0).item()
    return int(distance)

hash1 = "213e75503ac7113c6aeaf23d"
hash2 = "146db17232d8733e4cb916b4"

distance = hamming_distance(hash1, hash2)


print(f"The Hamming distance between the hashes is {distance}")