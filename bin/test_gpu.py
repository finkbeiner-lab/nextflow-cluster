import torch

print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print(f"GPUs detected: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print("Running a simple matrix multiplication on GPU...")

    # Perform matrix multiplication on GPU
    a = torch.randn(1000, 1000, device='cuda')
    b = torch.randn(1000, 1000, device='cuda')
    c = torch.matmul(a, b)

    print("Matrix multiplication successful on GPU.")
else:
    print("No GPU detected!")

