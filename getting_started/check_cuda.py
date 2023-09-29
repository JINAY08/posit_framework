import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available")

    # Check the installed CUDA version
    cuda_version = torch.version.cuda
    print("Installed CUDA version:", cuda_version)

    # Check PyTorch's CUDA compatibility
    pytorch_cuda_version = torch.version.cuda.split('.')
    pytorch_cuda_major = int(pytorch_cuda_version[0])
    pytorch_cuda_minor = int(pytorch_cuda_version[1])
    installed_cuda_major = int(cuda_version.split('.')[0])
    installed_cuda_minor = int(cuda_version.split('.')[1])

    if (pytorch_cuda_major == installed_cuda_major) and (pytorch_cuda_minor <= installed_cuda_minor):
        print("PyTorch is compatible with the installed CUDA version")
    else:
        print("PyTorch is not compatible with the installed CUDA version")
else:
    print("CUDA is not available")
