import numpy as np
import librosa
import os,sys,shutil,argparse,copy,pickle
from faceformer import Faceformer
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor

import torch
import ast

# template info
speaker_to_id = {
    "001Sky": 0,
    "002Shirley": 1,
    "003Alan": 2,
    "005Richard": 3,
    "006Vasilisa": 4,
    "007Jessica": 5,
    "008Kunio": 6
}
date_subjects = {
    "20231119": ["001Sky", "002Shirley"],
    "20231126": ["003Alan", "007Jessica"],
    "20231208": ["005Richard", "006Vasilisa"],
    "20240126": ["008Kunio", "006Vasilisa"],
    "20240128": ["001Sky", "007Jessica"]
}

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, default="prev_save/5-imp-mask/50_model.pth", help='path to the .pth model')
    parser.add_argument("--dataset", type=str, default="/data3/leoho/arfriend", help='base directory for dataset folder')
    parser.add_argument("--fps", type=float, default=30, help='frame rate')
    parser.add_argument("--period", type=int, default=30, help='period in PPE')
    parser.add_argument("--vertice_dim", type=int, default=17543*3, help='number of vertices - unmasked: 24049*3, masked: 17543*3')
    parser.add_argument("--feature_dim", type=int, default=128, help='feature dimensions')
    parser.add_argument("--render_template_path", type=str, default="templates", help='path of the template obj/ply mesh')
    parser.add_argument("--template_path", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--result_path", type=str, default="evals/faceformer-50", help='path of the predictions')
    parser.add_argument("--mask_path", type=str, default="mask/mask.pkl", help='path to the mask pickle file')
    parser.add_argument("--background_black", type=ast.literal_eval, default=True, help='whether to use black background')
    parser.add_argument("--train_subjects", type=str, default="001Sky 002Shirley")
    parser.add_argument("--eval_list", type=str, default="/data3/leoho/arfriend-diffspeaker/test_list_fair.txt")
    args = parser.parse_args()
    
    # limit cpu usage
    os.environ['OMP_NUM_THREADS'] = '8'
    torch.set_num_threads(8)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    # build model
    model = Faceformer(args)
    model.load_state_dict(torch.load(os.path.join(args.dataset, args.model_path)))
    model = model.to(torch.device(args.device))
    model.eval()
    
    # load eval list
    eval_targets = []
    with open(args.eval_list, 'r') as f:
         for line in f:
             eval_targets.append(line.strip())

    # load templates
    template_file = os.path.join(args.dataset, args.template_path)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
    
    # load unmasked templates 
    unmasked_template_file = '/data3/leoho/arfriend-diffspeaker/templates.pkl'
    with open(unmasked_template_file, 'rb') as fin:
        unmasked_templates = pickle.load(fin,encoding='latin1')
    
    # load mask file
    with open(os.path.join(args.dataset, args.mask_path), 'rb') as file:
        mask_dict = pickle.load(file)

    for eval in eval_targets:
        print(f'Predicting {eval}...')
        
        date_id = eval.split('_')[0]
        scenario_id = eval.split('_')[1]

        for actor in date_subjects[date_id]:
            # get template
            template = templates[actor]
            template = template.reshape((-1))
            template = np.reshape(template,(-1,template.shape[0]))
            template = torch.FloatTensor(template).to(device=args.device)
            
            # create one-hot vector
            one_hot_id = speaker_to_id[actor]
            one_hot = np.zeros(len(speaker_to_id))
            one_hot[one_hot_id] = 1
            one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
            one_hot = torch.FloatTensor(one_hot).to(device=args.device)
            
            # load audio
            wav_path = os.path.join(args.dataset, 'wav', date_id+'_'+actor+'_'+scenario_id+'.wav')
            speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            audio_feature = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
            audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
            audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)
            
            # make prediction
            prediction = model.predict(audio_feature, template, one_hot)
            prediction = prediction.squeeze() # (seq_len, V*3)
            prediction = prediction.detach().cpu().numpy()
            
            # merge prediction into unmasked
            unmasked_template = unmasked_templates[actor]
            unmasked_template = unmasked_template.reshape((-1))
            unmasked_template = np.reshape(unmasked_template,(-1,unmasked_template.shape[0]))
            result = np.repeat(unmasked_template, prediction.shape[0], axis=0)
            prediction_mask_verts = mask_dict['prediction_mask_verts']
            prediction_mask_verts = prediction_mask_verts * 3
            prediction_mask_verts = np.hstack([np.arange(i, i+3) for i in prediction_mask_verts])
            for i in range(prediction.shape[0]):
                result[i][prediction_mask_verts] = prediction[i]
            
            np.save(os.path.join(args.result_path, date_id+'_'+actor+'_'+scenario_id+'.npy'), result)
    
    print('All done!')

if __name__=="__main__":
    main()
