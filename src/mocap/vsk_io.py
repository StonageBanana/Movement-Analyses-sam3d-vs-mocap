"""Parse Vicon .vsk subject-anthropometrics files (PlugInGait model parameters)."""

import xml.etree.ElementTree as ET


def parse_vsk(path: str) -> dict:
    """Return {parameter_name: float}. Uses VALUE when present, else PRIOR."""
    tree = ET.parse(path)
    root = tree.getroot()
    params = {}
    for tag in ("StaticParameter", "Parameter"):
        for el in root.iter(tag):
            name = el.get("NAME")
            raw = el.get("VALUE") if el.get("VALUE") is not None else el.get("PRIOR")
            if name is None or raw is None:
                continue
            try:
                params[name] = float(raw)
            except ValueError:
                continue
    return params
