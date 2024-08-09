import numpy as np
import librosa
import os, argparse
from faceformer import Faceformer
from transformers import Wav2Vec2Processor
import torch
import ast

@torch.no_grad()
def test_model(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    #build model
    model = Faceformer(args)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(torch.device(args.device))
    model.eval()

    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    one_hot_labels = np.eye(len(train_subjects_list))
    iter = train_subjects_list.index(args.condition)
    one_hot = one_hot_labels[iter]
    one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
    one_hot = torch.FloatTensor(one_hot).to(device=args.device)

    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    audio_feature = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)

    prediction = model.predict(audio_feature, one_hot)
    prediction = prediction.squeeze() # (seq_len, V*3)
    np.save(os.path.join(args.result_path, test_name), prediction.detach().cpu().numpy())

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, default="save/150_model.pth", help='path to the .pth model')
    parser.add_argument("--dataset", type=str, default="data", help='base directory for dataset folder')
    parser.add_argument("--fps", type=float, default=30, help='frame rate')
    parser.add_argument("--period", type=int, default=30, help='period in PPE')
    parser.add_argument("--vertice_dim", type=int, default=55, help='number of vertices - unmasked: 24049*3, masked: 17543*3')
    parser.add_argument("--feature_dim", type=int, default=128, help='feature dimensions')
    parser.add_argument("--output_path", type=str, default="demo/output", help='path of the rendered video sequence')
    parser.add_argument("--wav_path", type=str, default="demo/wav/test.wav", help='path of the input audio signal')
    parser.add_argument("--result_path", type=str, default="demo/result", help='path of the predictions')
    parser.add_argument("--train_subjects", type=str, default="001Sky 002Shirley 003Alan 005Richard 006Vasilisa 007Jessica 008Kunio")
    parser.add_argument("--same_condition_as_subject", type=ast.literal_eval, default=True, help='whether to use the same conditioning subject as render subject')
    parser.add_argument("--subject", type=str, default="001Sky", help='select a subject from test_subjects or train_subjects')
    parser.add_argument("--condition", type=str, default="001Sky", help='select a conditioning subject from train_subjects')
    args = parser.parse_args()
    
    # limit cpu usage
    os.environ['OMP_NUM_THREADS'] = '8'
    torch.set_num_threads(8)  

    if args.same_condition_as_subject:
        args.condition = args.subject

    test_model(args)

if __name__=="__main__":
    main()
