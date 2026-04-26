from pathlib import Path
from tempfile import NamedTemporaryFile

import xacro


def expand_xacro(path, mappings, package_paths):
    text = Path(path).read_text()
    for name, package_path in package_paths.items():
        text = text.replace(f"$(find {name})", str(Path(package_path).resolve()))

    normalized = {
        key: str(value).lower() if isinstance(value, bool) else str(value)
        for key, value in mappings.items()
    }
    with NamedTemporaryFile("w", suffix=".xacro") as tmp:
        tmp.write(text)
        tmp.flush()
        xml = xacro.process_file(tmp.name, mappings=normalized).toxml()

    for name, package_path in package_paths.items():
        xml = xml.replace(f"package://{name}", Path(package_path).resolve().as_uri())
    return xml
