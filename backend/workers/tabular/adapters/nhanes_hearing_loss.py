"""Hearing loss (better-ear average above 25 dB) from four hearing questions. Trained by the Curanex team
(training/train_nhanes.py) on US CDC NHANES 2011-12, tested on 2015-16 (ages 20-69)."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_hearing_loss"
    target = "hearing_loss"
    input_schema = schema_for("hearing_loss")
    high = "Possible hearing loss"
    low = "Hearing loss less likely"
    note = "A hearing test (audiometry) confirms it. Sudden hearing loss in one ear needs a doctor within a day or two."

    def validate(self, inputs: dict) -> list[str]:
        errors = super().validate(inputs)
        if not errors and float(inputs["age"]) > 69:
            return ["age: this check was tested on people aged 20 to 69"]
        return errors
