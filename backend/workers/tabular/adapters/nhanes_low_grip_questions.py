"""Muscle weakness (handgrip below 27 kg men / 16 kg women, EWGSOP2) in people aged 50+, from simple questions. Trained by the
Curanex team on US CDC NHANES 2011-12; Verified on NHANES 2013-14 (sensitivity 0.821, AUC 0.831)."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_low_grip_questions"
    target = "low_grip_questions"
    input_schema = schema_for("low_grip_questions")
    high = "Possible muscle weakness"
    low = "Muscle weakness less likely"
    note = "A handgrip test at a clinic confirms it."
