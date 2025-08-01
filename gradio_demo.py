import gradio as gr
import predict
from predict import get_predict, get_predict_from_video


def detect_from_image(image):
    return get_predict(image)


def detect_from_video(video):
    return get_predict_from_video(video)



# İki ayrı tab ile resim ve video işleme
with gr.Blocks() as demo:
    gr.Markdown("# Nesne Tespiti - Resim ve Video")

    with gr.Tab("Resimden Tespit"):
        with gr.Row():
            image_input = gr.Image(type="numpy", label="Resim Yükle")
            image_output = gr.Image(label="Tespit Sonucu")

        image_button = gr.Button("Nesne Tespiti Yap")

        with gr.Row():
            table_output = gr.Dataframe(label="Tespit Tablosu")
            table2_output = gr.Dataframe(label="Uyarı Tablosu")

        with gr.Row():
            warn_str = gr.Textbox(label="Warning Notes")
        image_button.click(
            fn=detect_from_image,
            inputs=image_input,
            outputs=[image_output, table_output, table2_output, warn_str]
        )

    with gr.Tab("Videodan Tespit"):
        with gr.Row():
            video_input = gr.Video(label="Video Yükle")
            video_output = gr.Video(label="Tespit Sonucu")

        with gr.Row():
            video_warn_str = gr.Textbox(label="Warning Notes")

        with gr.Row():
            video_table_output = gr.Dataframe(label="Tespit Tablosu")

        video_button = gr.Button("Video İşle")
        video_button.click(
            fn=detect_from_video,
            inputs=video_input,
            outputs=[video_output, video_table_output, video_warn_str]
        )
if __name__ == "__main__":
    demo.launch()