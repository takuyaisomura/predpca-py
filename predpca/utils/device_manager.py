import multiprocessing as mp

import torch


class DeviceManager:
    def __init__(self):
        self.num_gpu = torch.cuda.device_count()
        manager = mp.Manager()
        self.locks = manager.list([manager.Lock() for _ in range(self.num_gpu)])

    def get_device(self) -> torch.device:
        if self.num_gpu == 0:
            return torch.device("cpu")
        for gpu_id in range(self.num_gpu):
            if self.locks[gpu_id].acquire(blocking=False):
                return torch.device(f"cuda:{gpu_id}")
        raise RuntimeError("No GPU available")

    def release_device(self, device: torch.device) -> None:
        if device.type == "cuda":
            self.locks[device.index].release()
