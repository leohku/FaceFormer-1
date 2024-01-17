import numpy as np
import scipy.io.wavfile as wav
import librosa
import os,sys,shutil,argparse,copy,pickle
import math,scipy
from faceformer import Faceformer
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import tempfile
from subprocess import call
os.environ['PYOPENGL_PLATFORM'] = 'osmesa' # egl
import pyrender
from psbody.mesh import Mesh
import trimesh
import math

@torch.no_grad()
def test_model(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    #build model
    model = Faceformer(args)
    model.load_state_dict(torch.load(os.path.join(args.dataset, args.model_path)))
    model = model.to(torch.device(args.device))
    model.eval()

    template_file = os.path.join(args.dataset, args.template_path)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')

    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    one_hot_labels = np.eye(len(train_subjects_list))
    iter = train_subjects_list.index(args.condition)
    one_hot = one_hot_labels[iter]
    one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
    one_hot = torch.FloatTensor(one_hot).to(device=args.device)

    temp = templates[args.subject]
             
    template = temp.reshape((-1))
    template = np.reshape(template,(-1,template.shape[0]))
    template = torch.FloatTensor(template).to(device=args.device)

    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    audio_feature = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)

    prediction = model.predict(audio_feature, template, one_hot)
    prediction = prediction.squeeze() # (seq_len, V*3)
    np.save(os.path.join(args.result_path, test_name), prediction.detach().cpu().numpy())

# The implementation of rendering is borrowed from VOCA: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
def render_mesh_helper(args, mesh, t_center, mask_dict, rot=np.zeros(3), tex_img=None, z_offset=0):
    camera_params = {'c': np.array([400, 400]),
                        'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                        'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center
    
    intensity = 2.0
    if args.pred_mask_color is None and args.importance_mask_color is None:
        rgb_per_v = None
        primitive_material = pyrender.material.MetallicRoughnessMaterial(
                    alphaMode='BLEND',
                    baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                    metallicFactor=0.8, 
                    roughnessFactor=0.8 
                )
    else:
        rgb_per_v = np.full((mesh.v.shape[0], mesh.v.shape[1]), [255, 255, 255])
        if args.pred_mask_color is not None:
            rgb_per_v[mask_dict['prediction_mask_verts']] = [int(i) for i in args.pred_mask_color.split()]
        if args.importance_mask_color is not None:
            rgb_per_v[mask_dict['importance_mask_verts']] = [int(i) for i in args.importance_mask_color.split()]
        primitive_material = None

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)

    if args.background_black:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1.2],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3,3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())
    
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except Exception as e:
        print('pyrender: Failed rendering frame: ' + str(e))
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def render_sequence(args):
    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    predicted_vertices_path = os.path.join(args.result_path,test_name+".npy")
    template_file = os.path.join(args.dataset, args.render_template_path, args.subject+'.obj')
    mask_dict = None
    if args.result_masked or args.pred_mask_color is not None or args.importance_mask_color is not None:
        with open(os.path.join(args.dataset, args.mask_path), 'rb') as file:
            mask_dict = pickle.load(file)
         
    print("rendering:", test_name, "with template:", args.subject+'.obj')
                 
    template = Mesh(filename=template_file)
    predicted_vertices = np.load(predicted_vertices_path)
    predicted_vertices = np.reshape(predicted_vertices,(-1,args.vertice_dim//3,3))

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    num_frames = predicted_vertices.shape[0]
    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=output_path)
    
    writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (800, 800), True)
    center = np.mean(template.v, axis=0)

    for i_frame in range(num_frames):
        if args.result_masked:
            prediction_mask_verts = mask_dict['prediction_mask_verts']
            template_v_copy = template.v.copy()
            template_v_copy[prediction_mask_verts] = predicted_vertices[i_frame]
            render_mesh = Mesh(template_v_copy, template.f)
        else:
            render_mesh = Mesh(predicted_vertices[i_frame], template.f)
        pred_img = render_mesh_helper(args, render_mesh, center, mask_dict)
        pred_img = pred_img.astype(np.uint8)
        writer.write(pred_img)

    writer.release()
    file_name = test_name+"_"+args.subject+"_condition_"+args.condition

    tmp_video_file_2 = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=output_path)
    cmd = ('ffmpeg' + ' -i {0} -pix_fmt yuv420p -qscale 0 -y {1}'.format(
       tmp_video_file.name, tmp_video_file_2.name)).split()
    call(cmd)

    # Add audio
    video_fname = os.path.join(output_path, file_name+'.mp4')
    cmd = ('ffmpeg' + ' -i {0} -i {1} -c:v copy -c:a flac -strict experimental -y {2}'.format(
         tmp_video_file_2.name, wav_path, video_fname)).split()
    call(cmd)

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, default="save/150_model.pth", help='path to the .pth model')
    parser.add_argument("--dataset", type=str, default="/data3/leoho/arfriend", help='base directory for dataset folder')
    parser.add_argument("--fps", type=float, default=30, help='frame rate')
    parser.add_argument("--period", type=int, default=30, help='period in PPE')
    parser.add_argument("--vertice_dim", type=int, default=17543*3, help='number of vertices - unmasked: 24049*3, masked: 17543*3')
    parser.add_argument("--feature_dim", type=int, default=128, help='feature dimensions')
    parser.add_argument("--render_template_path", type=str, default="templates", help='path of the template obj/ply mesh')
    parser.add_argument("--template_path", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--output_path", type=str, default="demo/output", help='path of the rendered video sequence')
    parser.add_argument("--wav_path", type=str, default="demo/wav/test.wav", help='path of the input audio signal')
    parser.add_argument("--result_path", type=str, default="demo/result", help='path of the predictions')
    parser.add_argument("--mask_path", type=str, default="mask/mask.pkl", help='path to the mask pickle file')
    parser.add_argument("--background_black", type=bool, default=True, help='whether to use black background')
    parser.add_argument("--result_masked", type=bool, default=True, help='whether the predictions are prediction masked')
    parser.add_argument("--train_subjects", type=str, default="001Sky 002Shirley")
    parser.add_argument("--same_condition_as_subject", type=bool, default=True, help='whether to use the same conditioning subject as render subject')
    parser.add_argument("--subject", type=str, default="001Sky", help='select a subject from test_subjects or train_subjects')
    parser.add_argument("--condition", type=str, default="001Sky", help='select a conditioning subject from train_subjects')
    parser.add_argument("--pred_mask_color", type=str, help='color of the mask area in "# # #", where they correspond to R,G,B values from 0-255')
    parser.add_argument("--importance_mask_color", type=str, help='color of the mask area in "# # #", where they correspond to R,G,B values from 0-255')
    args = parser.parse_args()
    
    # limit cpu usage
    os.environ['OMP_NUM_THREADS'] = '8'
    torch.set_num_threads(8)  

    if args.same_condition_as_subject:
        args.condition = args.subject

    test_model(args)
    render_sequence(args)

if __name__=="__main__":
    main()
