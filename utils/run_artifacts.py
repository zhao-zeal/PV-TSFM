"""Reserve experiment directories before any training or prediction writes."""

from pathlib import Path


def reserve_run(args, setting):
    if not setting or setting in ('.', '..') or Path(setting).name != setting:
        raise ValueError('model_id must be a single directory name')
    output = Path(args.output_dir) / args.protocol / setting
    checkpoint = Path(args.checkpoints) / args.protocol / setting
    if output.resolve() == checkpoint.resolve():
        raise ValueError('output_dir and checkpoints must be different directories')
    for path in (output, checkpoint):
        if path.exists():
            raise FileExistsError(f'Run already exists: {path}; use a new --model_id, including for resume/evaluation')
    created = []
    try:
        for path in (checkpoint, output):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.mkdir()  # Exclusive creation also rejects concurrent use of one ID.
            created.append(path)
    except OSError:
        for path in reversed(created):
            path.rmdir()  # Only our still-empty reservations are removed.
        raise
    return output
