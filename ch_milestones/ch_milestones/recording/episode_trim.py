from pathlib import Path


class InsertionEventTrim:
    def __init__(self, node, post_frames: int, require_event: bool):
        self.node = node
        self.post_frames = post_frames
        self.require_event = require_event
        self.frame = None
        self.port = ""

    def mark(self, frame: int, port: str):
        if self.frame is None:
            self.frame = frame
            self.port = port

    def apply(self, dataset):
        if self.frame is None:
            if self.require_event:
                raise RuntimeError("No /scoring/insertion_event received")
            return
        buffer = dataset.writer.episode_buffer
        size = buffer["size"]
        keep = min(size, self.frame + self.post_frames)
        if keep >= size:
            return
        self.delete_visual_frames(dataset, buffer, keep)
        for key, value in buffer.items():
            if isinstance(value, list):
                buffer[key] = value[:keep]
        buffer["size"] = keep
        self.node.get_logger().info(
            f"Trimmed episode from {size} to {keep} frames after insertion at {self.port}"
        )

    def delete_visual_frames(self, dataset, buffer, keep):
        for key in dataset.meta.camera_keys:
            for path in buffer[key][keep:]:
                Path(path).unlink()
