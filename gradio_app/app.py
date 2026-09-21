"""Gradio demo for vehicle re-identification."""

import gradio as gr

from reid_inference import DEFAULT_CHECKPOINT, ReIDEmbedder


embedder = ReIDEmbedder(DEFAULT_CHECKPOINT)


def compare_images(reference_image, query_image, threshold):
    if reference_image is None or query_image is None:
        return "Add both images to run the comparison.", "", ""

    result = embedder.compare(reference_image, query_image, threshold)

    if result["needs_review"]:
        decision = "NEEDS REVIEW"
    elif result["same_vehicle"]:
        decision = "SAME VEHICLE"
    else:
        decision = "DIFFERENT VEHICLES"

    plate_status = result["plate_status"].upper()
    details = (
        f"Cosine similarity: {result['similarity']:.4f}\n"
        f"Threshold: {result['threshold']:.2f}\n"
        f"Plate status: {plate_status}\n"
        f"Device: {result['device']}\n"
        f"Flags: {'; '.join(result['flags']) if result['flags'] else 'None'}"
    )
    return decision, plate_status, details


with gr.Blocks(title="Vehicle Re-Identification") as demo:
    gr.Markdown("# Vehicle Re-Identification\nCompare two vehicle images with the trained ResNet-50 model and plate safeguard.")
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
    plate_status = gr.Textbox(label="Plate status", interactive=False)
    details = gr.Textbox(label="Details", lines=5, interactive=False)
    compare_button.click(
        compare_images,
        inputs=[reference_image, query_image, threshold],
        outputs=[decision, plate_status, details],
    )


if __name__ == "__main__":
    demo.launch(share=True)