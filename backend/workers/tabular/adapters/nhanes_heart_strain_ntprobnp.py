"""Heart strain: raised NT-proBNP (125 pg/mL or more, the ESC heart failure rule-out cut-off) in adults, from simple questions.
Trained by the Curanex team on US CDC NHANES 1999-2002 (training/train_nhanes_p.py); Verified on NHANES 2003-04
(sensitivity 0.868, AUC 0.878)."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_heart_strain_ntprobnp"
    target = "heart_strain_ntprobnp"
    input_schema = schema_for("heart_strain_ntprobnp")
    high = "Possible heart strain"
    low = "Heart strain less likely"
    note = "An NT-proBNP blood test and an ECG or echo confirm it. Chest pain or breathlessness now needs urgent care: call 112."
