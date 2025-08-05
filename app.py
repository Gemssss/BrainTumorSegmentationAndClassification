import gradio as gr
import time
from PIL import Image
from utils.inference import predict_and_segment

# === Fake Loading Bar ===
def fake_loading():
    for i in range(0, 101, 10):
        bar = "█" * (i // 10) + "░" * (10 - i // 10)
        yield f"## Analyzing Image...\n`{bar}` {i}%"
        time.sleep(0.3)

# === Model Prediction Function ===
def run_model(image):
    label, conf, output_img = predict_and_segment(image)
    label_text = f"**Tumor Type:** {label}  \n**Confidence:** {conf*100:.2f}%"
    return label_text, output_img, image

# === MAIN APP ===
with gr.Blocks(title="Brain Tumor Detection") as app:
    # PAGE 1: Upload Page
    with gr.Column(visible=True, elem_id="page1") as page1:
        gr.Markdown("## Brain Tumor Detection", elem_id="title", elem_classes="center")
        image_input = gr.Image(
            type="pil", 
            label="Upload MRI Image", 
            sources=["upload"],
            show_label=False,
        )
        analyze_btn = gr.Button("Run The Analysis")

    # PAGE 2: Centered Loading Page
    with gr.Row(visible=False, elem_id="page2") as page2:
        with gr.Column(scale=1, min_width=0):
            pass  # empty left spacer
        with gr.Column(scale=2, min_width=0):
            loading_output = gr.Markdown("## Analyzing Image...\n`░░░░░░░░░░` 0%", elem_id="loading-bar", elem_classes="center-text")
        with gr.Column(scale=1, min_width=0):
            pass  # empty right spacer

    # PAGE 3: Result Page
    with gr.Column(visible=False, elem_id="page3") as page3:
        with gr.Row(elem_classes="center-row"):
            original_image_output = gr.Image(label="Original Uploaded Image")
        with gr.Row(elem_classes="center-row"):
            segmented_image_output = gr.Image(label="Segmented Image")
        with gr.Row(elem_classes="center-row"):
            label_output = gr.Markdown(elem_classes="center-text")
        with gr.Row(elem_classes="center-row"):
            rerun_btn = gr.Button("🔄 Re-run Analysis", size="lg")


    # BUTTON LOGIC
    analyze_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)),
        inputs=[],
        outputs=[page1, page2, page3]
    ).then(
        fn=fake_loading,
        inputs=[],
        outputs=loading_output
    ).then(
        fn=run_model,
        inputs=image_input,
        outputs=[label_output, segmented_image_output, original_image_output]
    ).then(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        inputs=[],
        outputs=[page2, page3]
    )

    # RERUN RESET
    rerun_btn.click(
        fn=lambda: (None, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
        inputs=[],
        outputs=[image_input, page1, page2, page3]
    )

# === LAUNCH ===
if __name__ == "__main__":
    app.launch()
