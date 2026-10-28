"""
app.py  (M4 — Gradio demo)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO EXPLAIN THIS MODULE (Day-6 dry run):
  • What it does : A ~60-line Gradio app: target post + optional parent, shows BOTH class
                   probabilities, latency (ms), test Macro-F1, top-3 token attributions, and a
                   "load a failure" button that loads an example the model gets WRONG.
  • Inputs       : user text; loads best_model/ at startup (weights downloaded locally — never
                   depend on HF Hub at demo time).
  • Outputs      : live prediction in the browser at http://localhost:7860.
  • The one gotcha: record a 90-second backup video. A video that always works beats a live Space
                   that cold-starts for 30–60s when the examiner clicks the link.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import gradio as gr


def predict(target: str, parent: str = "", use_context: bool = True):
    # PLACEHOLDER: reuse src/predict.py. Return {HATE: p, NON-HATE: 1-p}, latency_ms, attributions.
    return {"HATE": 0.5, "NON-HATE": 0.5}, 340.0, ["<top-3 tokens>"]


def load_failure_example():
    return "bhai ye kis level ki ghatiya soch hai", ""  # masked slur in real use


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Target post", placeholder="bhai ye kis level ki ghatiya soch hai"),
        gr.Textbox(label="Parent post (optional context)", placeholder="..."),
        gr.Checkbox(label="Use context", value=True),
    ],
    outputs=[
        gr.Label(label="Prediction"),
        gr.Number(label="Latency (ms)"),
        gr.JSON(label="Top-3 attributions"),
    ],
    title="Code-Mixed Hinglish Hate Speech Detection (A0: MuRIL + context)",
    examples=[load_failure_example()],
)

if __name__ == "__main__":
    demo.launch()
