"""Gradio demo for vehicle re-identification."""

import gradio as gr

from reid_inference import DEFAULT_CHECKPOINT, ReIDEmbedder


embedder = ReIDEmbedder(DEFAULT_CHECKPOINT)


def compare_images(reference_image, query_image, threshold):
    if reference_image is None or query_image is None:
        return "Add both images to run the comparison.", ""

    result = embedder.compare(reference_image, query_image, threshold)
    decision = "SAME VEHICLE" if result["same_vehicle"] else "DIFFERENT VEHICLES"
    details = (
        f"Cosine similarity: {result['similarity']:.4f}\n"
        f"Threshold: {result['threshold']:.2f}\n"
        f"Device: {result['device']}"
    )
    return decision, details


with gr.Blocks(title="Vehicle Re-Identification") as demo:
    gr.Markdown("# Vehicle Re-Identification\nCompare two vehicle images with the trained ResNet-50 model.")
    with gr.Row():
        reference_image = gr.Image(type="pil", label="Reference image")
        query_image = gr.Image(type="pil", label="Query image")
    threshold = gr.Slider(
        minimum=-1.0,
        maximum=1.0,
        value=0.70,
        step=0.01,
        label="Similarity threshold",
    )
    compare_button = gr.Button("Compare", variant="primary")
    decision = gr.Textbox(label="Decision", interactive=False)
    details = gr.Textbox(label="Details", lines=3, interactive=False)
    compare_button.click(
        compare_images,
        inputs=[reference_image, query_image, threshold],
        outputs=[decision, details],
    )


if __name__ == "__main__":
    demo.launch(share=True)