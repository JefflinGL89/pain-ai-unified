import json
from pathlib import Path

out=Path("data/project_results.json")
template={
 "metrics":{"accuracy":0.7843,"balanced_accuracy":0.5659,"macro_f1":0.5944},
 "dataset":{"frames":48398,"subjects":25,"sequences":200,"pspi_mean":0.446981,"pspi_std":1.302021,"pspi_min":0,"pspi_max":15},
 "class_distribution":[],"confusion_matrix":[],"ordinal_error":[],"pspi_accuracy":[],"confidence_distribution":[],"subject_accuracy":[],"training_loss":[]
}
out.write_text(json.dumps(template,indent=2))
print("Created",out)
