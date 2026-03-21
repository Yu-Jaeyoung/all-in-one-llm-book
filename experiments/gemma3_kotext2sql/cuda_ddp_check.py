import os
import torch
import torch.distributed as dist

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    x = torch.ones(1, device=f"cuda:{local_rank}")
    dist.all_reduce(x)
    print(f"rank={dist.get_rank()} local_rank={local_rank} x={x.item()}")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
