import os
import socket
import torch

def initialize_distributed(args):
    """Initialize torch.distributed."""
    if torch.distributed.is_initialized():
        return True

    # Single-process runs should not initialize distributed.
    if getattr(args, "world_size", 1) == 1:
        args.rank = 0
        return False
    # the automatic assignment of devices has been moved to arguments.py
    if args.device == "cpu":
        pass
    else:
        if isinstance(args.device, str) and args.device.startswith("npu"):
            import torch_npu  # noqa: F401
            idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
            torch.npu.set_device(idx)
        elif isinstance(args.device, str) and args.device.startswith("cuda"):
            idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
            torch.cuda.set_device(idx)
        else:
            # fallback: try cuda
            try:
                torch.cuda.set_device(args.device)
            except Exception:
                pass
    # Call the init process
    init_method = "tcp://"
    args.master_ip = os.getenv("MASTER_ADDR", "localhost")

    if args.world_size == 1:
        default_master_port = str(get_free_port())
    else:
        default_master_port = "6000"
    args.master_port = os.getenv("MASTER_PORT", default_master_port)
    if args.world_size == 1 and is_port_in_use(args.master_port, args.master_ip):
        args.master_port = str(get_free_port())
    init_method += args.master_ip + ":" + args.master_port
    print(f'| rank {args.rank} distribution init at {init_method}')

    init_method = "env://" #暂时这样使用
    backend = 'nccl'
    if isinstance(args.device, str) and args.device.startswith('npu'):
        backend = 'hccl'
    elif args.device == 'cpu':
        backend = 'gloo'
    torch.distributed.init_process_group(
        backend=backend, world_size=args.world_size, rank=args.rank, init_method=init_method
    )
    return True

def is_port_in_use(port, host='127.0.0.1'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, int(port)))
            return False  # 端口未被占用
        except OSError:
            return True  # 端口被占用
        
def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # 0 表示由系统自动分配端口
        return s.getsockname()[1]
