# scripts/per_class_gain.py
import pandas as pd

piso = pd.read_csv("runs/resnet18_b5_s0/test_per_class.csv")[["class_name", "f1"]]
aluno = pd.read_csv("runs/student_perclass_resnet18_b5_s2/test_per_class.csv")[["class_name", "f1"]]
m = piso.merge(aluno, on="class_name", suffixes=("_piso", "_aluno"))
m["delta_f1"] = m["f1_aluno"] - m["f1_piso"]
m = m.sort_values("delta_f1", ascending=False)
print(m.to_string(index=False))
m.to_csv("per_class_gain_b5.csv", index=False)