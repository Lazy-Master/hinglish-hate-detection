"""
app.py  (M4 — Gradio demo)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO EXPLAIN THIS MODULE (Day-6 dry run):
  • What it does : A ~60-line Gradio app: a single post in, BOTH class probabilities, latency (ms),
                   test Macro-F1, top-3 token attributions, and a "load a failure" example.
  • Inputs       : user text; loads best_model/ at startup (weights downloaded locally — never
                   depend on HF Hub at demo time).
  • Outputs      : live prediction in the browser at http://localhost:7860.
  • The one gotcha: record a 90-second backup video. A video that always works beats a live Space
                   that cold-starts for 30–60s when the examiner clicks the link.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import gradio as gr


def predict(text: str):
    # PLACEHOLDER: reuse src/predict.py. Return {HATE: p, NON-HATE: 1-p}, latency_ms, attributions.
    return {"HATE": 0.5, "NON-HATE": 0.5}, 340.0, ["<top-3 tokens>"]


def load_failure_example():
    return "bhai ye kis level ki ghatiya soch hai"  # masked slur in real use


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Post", placeholder="bhai ye kis level ki ghatiya soch hai"),
    ],
    outputs=[
        gr.Label(label="Prediction"),
        gr.Number(label="Latency (ms)"),
        gr.JSON(label="Top-3 attributions"),
    ],
    title="Code-Mixed Hinglish Hate Speech Detection (A0: MuRIL)",
    examples=[load_failure_example()],
)

if __name__ == "__main__":
    demo.launch()
