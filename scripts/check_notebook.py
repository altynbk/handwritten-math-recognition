"""Execute all walkthrough cells, saving the result only under ignored runs/."""
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "notebooks/walkthrough.ipynb"
destination = ROOT / "runs/walkthrough.executed.ipynb"
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
# Capture plots in notebook outputs even when the runner defaults to headless Agg.
os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
notebook = nbformat.read(source, as_version=4)
client = NotebookClient(notebook, timeout=300, kernel_name="python3",
                        resources={"metadata": {"path": str(ROOT)}})
# Start the same interpreter as this script, regardless of installed user kernels.
client.km = client.create_kernel_manager()
client.km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client.execute()
plots = sum("image/png" in output.get("data", {})
            for cell in notebook.cells for output in cell.get("outputs", []))
if plots < 5:
    raise RuntimeError(f"Expected the walkthrough's five plots, but captured only {plots}.")
destination.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, destination)
print(f"Executed {sum(cell.cell_type == 'code' for cell in notebook.cells)} code cells "
      f"and captured {plots} plots: {destination}")
