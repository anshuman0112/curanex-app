"""Test model for the pipeline (not medical). Turns echo_score 0-10 into a 'Higher'/'Lower' probability."""
from workers.common.adapter import ModelAdapter, Prediction


class Adapter(ModelAdapter):
    id = "dummy_echo"
    input_schema = {
        "type": "object",
        "required": ["echo_score", "age"],
        "properties": {
            "echo_score": {"type": "number", "minimum": 0, "maximum": 10},
            "age": {"type": "integer", "minimum": 0, "maximum": 120},
        },
    }

    def predict(self, inputs: dict) -> Prediction:
        p = float(inputs["echo_score"]) / 10
        return Prediction(model_id=self.id, label="Higher" if p >= 0.5 else "Lower",
                          probabilities={"Lower": round(1 - p, 3), "Higher": round(p, 3)},
                          explanation={"top_factors": [["echo_score", round(p - 0.5, 3)]]})
