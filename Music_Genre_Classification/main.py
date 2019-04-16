from load_model_for_style_transfer import load_model_for_style_transfer
from preprocess_audio import preprocess_audio
from style_transfer import style_transfer

if __name__ == '__main__':
    # Preprocess Audio Data
    content_audio, style_audio = preprocess_audio()

    # Load Model
    model = load_model_for_style_transfer()

    # Start Style Transfer
    style_transfer(model, content_audio, style_audio)
