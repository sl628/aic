import time

import rclpy


def call(node, client, request, timeout_sec: float, label: str):
    if not client.wait_for_service(timeout_sec=timeout_sec):
        raise TimeoutError(f"{label} service is not available")
    future = client.call_async(request)
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok() and not future.done() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not future.done():
        raise TimeoutError(f"Timed out waiting for {label}")
    return future.result()
