import torch
import torchaudio
from safetensors.torch import load_file
from parler_tts import DACModel

tensors = load_file("out.safetensors")
dac_model = DACModel.from_pretrained("parler-tts/dac_44khZ_8kbps")
print(dac_model.model)
output_ids = tensors["codes"][None, None]
print(output_ids, "\n", output_ids.shape)
batch_size = 1
with torch.no_grad():
    output_values = []
    for sample_id in range(batch_size):
        sample = output_ids[:, sample_id]
        sample_mask = (sample >= dac_model.config.codebook_size).sum(dim=(0, 1)) == 0
        if sample_mask.sum() > 0:
            sample = sample[:, :, sample_mask]
            sample = dac_model.decode(sample[None, ...], [None]).audio_values
            output_values.append(sample.transpose(0, 2))
        else:
            output_values.append(torch.zeros((1, 1, 1)).to(dac_model.device))
    output_lengths = [audio.shape[0] for audio in output_values]
    pcm = (
        torch.nn.utils.rnn.pad_sequence(output_values, batch_first=True, padding_value=0)
        .squeeze(-1)
        .squeeze(-1)
    )
print(pcm.shape, pcm.dtype)
torchaudio.save("out.wav", pcm.cpu(), sample_rate=44100)
