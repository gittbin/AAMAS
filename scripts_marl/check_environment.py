"""Print local capabilities before starting paper experiments."""
import importlib
import json
import shutil
import sys
report = {"python": sys.version, "executable": sys.executable}
for name in ("torch", "numpy", "scipy", "gymnasium", "pettingzoo", "yaml", "pytest"):
    try:
        module = importlib.import_module(name)
        report[name] = getattr(module, "__version__", "installed")
        if name == "torch":
            report["cuda_available"] = module.cuda.is_available()
            report["gpu"] = module.cuda.get_device_name(0) if module.cuda.is_available() else None
    except ImportError as exc:
        report[name] = str(exc)
for name in ("conda", "pdflatex", "bibtex", "nvidia-smi"):
    report[name] = shutil.which(name)
print(json.dumps(report, indent=2))
