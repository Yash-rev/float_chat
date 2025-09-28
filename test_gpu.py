import torch
if torch.cuda.is_available():
    print(f"GPU is available!")
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPU(s).")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available. PyTorch is using the CPU.")