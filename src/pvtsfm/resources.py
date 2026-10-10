import subprocess


def require_idle_gpus(gpus):
    selected = gpus.split(',')
    if not 1 <= len(selected) <= 4 or len(set(selected)) != len(selected):
        raise ValueError('Select 1–4 distinct idle GPUs')
    devices = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True,
    )
    processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True,
    ).splitlines()
    gpu_info = {row[0].strip(): [x.strip() for x in row[1:]]
                for row in (line.split(',') for line in devices.splitlines())}
    for index in selected:
        if index not in gpu_info:
            raise ValueError(f'GPU {index} does not exist')
        uuid, memory, utilization = gpu_info[index]
        if uuid in processes or int(memory) > 1024 or int(utilization) > 0:
            raise ValueError(f'GPU {index} is occupied; select an idle GPU')
