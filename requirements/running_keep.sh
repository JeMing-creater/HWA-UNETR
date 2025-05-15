export OMP_NUM_THREADS=1
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

PORT=$(python -c "import socket;s = socket.socket(socket.AF_INET, socket.SOCK_STREAM);s.bind(('', 0));port = s.getsockname()[1];s.close();print(port)")

tee /etc/feed_gpu.py <<-'EOF'
import torch
from accelerate import Accelerator

if __name__ == "__main__":
    model = torch.nn.Linear(1, 1)
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    print(model)
    input()

EOF

torchrun \
  --nproc_per_node ${NUM_GPUS} \
  --master_port ${PORT} \
  /etc/feed_gpu.py