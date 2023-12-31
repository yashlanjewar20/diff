import gradio as gr
import torch
import torchaudio
import os
import time
from scipy.io import wavfile

from resemble_enhance.enhancer.inference import denoise, enhance


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
os.environ['GRADIO_TEMP_DIR'] = '/home/yash/temp'

def _fn(path, solver, nfe, tau, denoising, run_mode = "fp_16"):
    if path is None:
        return None, None

    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoising else 0.1

    dwav, sr = torchaudio.load(path)
    dwav = dwav.mean(dim=0)
    
    
    dwav = dwav.to(device).to(torch.float32)
    
    start_time = time.time()

    wav1, new_sr = denoise(dwav, sr, device, run_mode)
    wav2, new_sr = enhance(run_mode, dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)

    end_time= time.time()
    
    duration = end_time - start_time
    #print(f"Processing time: {duration} seconds")

    
    wav1 = wav1.cpu().numpy()
    wav2 = wav2.cpu().numpy()

    return (new_sr, wav1), (new_sr, wav2)


def enhance_using_resemble(path, out_path, solver='Euler', nfe=1, tau=0.5, denoising=False, run_mode = "fp_16"):
    """
        This function is used to enhance the audio using resemble's pretrained model. The default values of the parameters are set to best values for efficient and fast processing.
    """
    if path is None:
        return None, None

    # setting the parameters
    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoising else 0.1
    
    # loading the audio file
    dwav, sr = torchaudio.load(path)
    dwav = dwav.mean(dim=0)
    dwav = dwav.to(device).to(torch.float32)
    
    # enhancing the audio
    if(denoising):
        wav1, new_sr = denoise(dwav, sr, device, run_mode)
    wav2, new_sr = enhance(run_mode, dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)
    wav2 = wav2.cpu().numpy()
    
    # save the wav to file
    wavfile.write(out_path, new_sr, wav2)

    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to input audio file")
    parser.add_argument("--out_path", type=str, help="Path to output audio file")
    args = parser.parse_args()
    
    enhance_using_resemble(args.path, args.out_path)
