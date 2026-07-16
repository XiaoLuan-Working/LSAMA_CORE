import torch
import torch.nn.functional as F
import numpy as np

def calculate_2d_entropy(image_tensor, num_bins=128):
    device = image_tensor.device
    B, C, H, W = image_tensor.shape

    if H > 64 or W > 64:
        image_tensor = F.interpolate(image_tensor, size=(64, 64), mode='bilinear', align_corners=False)
        B, C, H, W = image_tensor.shape

    img_scaled = image_tensor * (num_bins - 1)

    kernel = torch.ones((C, 1, 3, 3), device=device) / 9.0
    neighborhood_mean = F.conv2d(img_scaled, kernel, padding=1, groups=C)

    bins = torch.arange(num_bins, dtype=torch.float32, device=device).view(1, num_bins, 1, 1)

    sigma = 1.0
    weights_i = torch.exp(-((img_scaled.unsqueeze(1) - bins)**2) / (2 * sigma**2))
    weights_i = weights_i / (torch.sum(weights_i, dim=1, keepdim=True) + 1e-8)

    weights_j = torch.exp(-((neighborhood_mean.unsqueeze(1) - bins)**2) / (2 * sigma**2))
    weights_j = weights_j / (torch.sum(weights_j, dim=1, keepdim=True) + 1e-8)

    wi_flat = weights_i.view(B, num_bins, 1, -1)
    wj_flat = weights_j.view(B, 1, num_bins, -1)

    joint_hist = torch.sum(wi_flat * wj_flat, dim=-1)

    total_pixels = H * W
    P_ij = joint_hist / total_pixels

    P_ij = P_ij + 1e-8

    P_i = torch.sum(P_ij, dim=2)
    P_j = torch.sum(P_ij, dim=1)

    entropy_2d = -torch.sum(P_ij * torch.log(P_ij), dim=(1, 2))

    feature_descriptor = {
        'entropy_2d': entropy_2d,
        'P_i_mean': P_i.mean(dim=1),
        'P_i_var': P_i.var(dim=1),
    }

    return feature_descriptor

def compute_distribution_loss(orig_img_tensor, prime_img_tensor, Delta=0.1):
    orig_features = calculate_2d_entropy(orig_img_tensor)
    prime_features = calculate_2d_entropy(prime_img_tensor)

    loss_entropy = torch.abs(prime_features['entropy_2d'] - orig_features['entropy_2d'])

    loss_var = torch.abs(prime_features['P_i_var'] - (1.0 + Delta) * orig_features['P_i_var'])

    L_dist = loss_entropy.mean() + loss_var.mean()

    return L_dist

if __name__ == "__main__":
    print("Testing 2D-Entropy distribution feature extractor (differentiable)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dummy_img = torch.rand(1, 1, 64, 64, device=device, requires_grad=True)
    prime_img = dummy_img * 0.9 + 0.05

    L_dist = compute_distribution_loss(dummy_img, prime_img)

    L_dist.backward()

    print(f"Test passed! L_dist = {L_dist.item():.4f}")
    print(f"Gradient backpropagation works (has valid grad: {dummy_img.grad is not None}).")
