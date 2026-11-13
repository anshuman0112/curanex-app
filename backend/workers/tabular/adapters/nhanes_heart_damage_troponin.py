"""Raised heart-muscle troponin (high-sensitivity troponin T 14 ng/L or more, the 99th-percentile limit) in adults, from simple
questions. Trained by the Curanex team on US CDC NHANES 1999-2002 (training/train_nhanes_q.py); Verified on NHANES 2003-04
(sensitivity 0.856, AUC 0.905). It estimates long-term heart strain in people without symptoms; it is not for a heart attack happening now."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_heart_damage_troponin"
    target = "heart_damage_troponin"
    input_schema = schema_for("heart_damage_troponin")
    high = "Possible raised heart troponin"
    low = "Raised heart troponin less likely"
    note = "Not for chest pain happening now: chest pain, sweating or breathlessness needs urgent care, call 112. A troponin blood test confirms it."
