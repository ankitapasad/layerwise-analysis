import numpy as np
import os
import random
import sys

import fairseq
import soundfile as sf
import torch
import torch.nn.functional as F

from prepare_utils import PER_LAYER_TRANSFORM_DCT

curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(curr_dir, ".."))
from utils import load_dct


class ModelLoader:
    def __init__(self, ckpt_pth, model_type, pckg_dir=None, dict_fn=None):
        """
        load model from ckpt_pth
        """
        self.ckpt_pth = ckpt_pth
        self.model_type = model_type
        self.pckg_dir = pckg_dir
        self.dict_fn = dict_fn

    def wavlm(self):
        sys.path.insert(0, self.pckg_dir)
        from WavLM import WavLM, WavLMConfig

        ckpt = torch.load(self.ckpt_pth)
        cfg = WavLMConfig(ckpt["cfg"])
        encoder = WavLM(cfg)
        encoder.load_state_dict(ckpt["model"])
        return encoder, cfg

    def avhubert(self):
        from argparse import Namespace
        ckpt_dct = torch.load(self.ckpt_pth)
        audio_branch_keys = ['layer_norm', 'post_extract_proj', 'encoder', 'feature_extractor_audio', 'final_proj', 'mask_emb', 'label_embs_concat']
        other_keys = ['feature_extractor_video']
        # audio_branch_keys = ['encoder', 'feature_extractor_audio']
        # other_keys = ['feature_extractor_video', 'layer_norm', 'post_extract_proj', 'final_proj', 'mask_emb', 'label_embs_concat']
        self.cnt_num_params(ckpt_dct, audio_branch_keys, other_keys, main_key='model')
        fairseq.utils.import_user_module(Namespace(user_dir=self.pckg_dir))
        models, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.ckpt_pth]
        )
        encoder = models[0]
        return encoder, task.cfg

    def s3adapt(self):
        import s3prl.hub as hub
        model = getattr(hub, "hubert")()
        ckpt_dct = torch.load(self.ckpt_pth)
        ckpt_wt = ckpt_dct.get("Upstream")
        model.load_state_dict(ckpt_wt)
        return model, model.task_cfg

    def fairseq_model_loader(self):
        if self.model_type == "finetuned":
            assert self.dict_fn
            model, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [self.ckpt_pth],
                arg_overrides={"data": self.dict_dn},
            )
            model = model[0]
            encoder = model.w2v_encoder._modules["w2v_model"]
        else:
            (
                encoder,
                _,
                task,
            ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.ckpt_pth])
            encoder = encoder[0]
        task_cfg = task.cfg
        return encoder, task_cfg

    def wavembed(self):
        sys.path.append(self.pckg_dir)
        from WavEmbed import WavEmbedModel
        model = WavEmbedModel.from_pretrained("charsiu/WavEmbed_100")
        return model, -1
    
    def shubert(self):
        sys.path.append(self.pckg_dir)
        from SentHuBERT import SentHuBERT
        model = SentHuBERT.from_pretrained('charsiu/S-HuBERT-from-simcse-sup-roberta')
        return model, -1

    def wav2vec(self):
        encoder, task_cfg = self.fairseq_model_loader()
        return encoder, task_cfg

    def xlsr53(self):
        encoder, task_cfg = self.fairseq_model_loader()
        return encoder, task_cfg

    def xlsr128(self):
        encoder, task_cfg = self.fairseq_model_loader()
        return encoder, task_cfg

    def hubert(self):
        encoder, task_cfg = self.fairseq_model_loader()
        return encoder, task_cfg

    def data2vec(self):
        def load_cfg(cfg_obj, args_dct):
            for key, value in args_dct.items():
                if hasattr(cfg_obj, key):
                    setattr(cfg_obj, key, value)

        def load_model(encoder, args_dct):
            del args_dct["_ema"]
            args_dct["final_proj.bias"] = args_dct["final_proj.0.bias"] 
            args_dct["final_proj.weight"] = args_dct["final_proj.0.weight"] 
            del args_dct["final_proj.0.bias"]
            del args_dct["final_proj.0.weight"]
            encoder.load_state_dict(args_dct)

        sys.path.append(self.pckg_dir)
        from data2vec_audio import  Data2VecAudioConfig, Data2VecAudioModel
        ckpt = torch.load(self.ckpt_pth)

        cfg = Data2VecAudioConfig
        load_cfg(cfg, ckpt['cfg']['model'])
        # cfg = Data2VecAudioConfig(**ckpt["cfg"]["model"]) # does not work coz of extra keys
        encoder = Data2VecAudioModel(cfg)
        load_model(encoder, ckpt["model"])
        return encoder, ckpt["cfg"]["task"]

    def cnt_num_params(self, weights, audio_branch_keys, other_keys, main_key='dual_encoder'):
        """
        count number of parameters in the audio branch alone
        """
        all_keys = weights[main_key].keys()
        num_params = 0
        unused_keys = []
        for key in all_keys:
            if any([key.startswith(audio_branch_key) for audio_branch_key in audio_branch_keys]):
                num_params += weights[main_key][key].numel()
            else:
                try:
                    assert any([key.startswith(other_key) for other_key in other_keys])
                except:
                    import pdb; pdb.set_trace()
                unused_keys.append(key)
        print(num_params // 1e6, 'M params in audio branch')
        # import pdb; pdb.set_trace()
    
    def fastvgs(self):
        sys.path.append(self.pckg_dir)
        from models import w2v2_model
        from models import fast_vgs_edit as fast_vgs

        audio_branch_keys = ['conv1', 'trm1', 'trm3', 'conv2', 'trm2', 'audio_cls']
        other_keys = ['visual', 'visn', 'trm.']
        weights = torch.load(f"{self.ckpt_pth}/best_bundle.pth")
        self.cnt_num_params(weights, audio_branch_keys, other_keys, main_key='dual_encoder')

        args = load_dct(f"{self.ckpt_pth}/args.pkl")

        dual_encoder = fast_vgs.DualEncoder(args)
        dual_encoder.load_state_dict(weights["dual_encoder"])

        model = w2v2_model.Wav2Vec2Model_cls(args)
        model.carefully_load_state_dict(
            weights["dual_encoder"]
        )  # will filter out weights that don't belong to w2v2
        return [model, dual_encoder], args

    def fastvgs_coco(self):
        self.fastvgs()

    def fastvgs_places(self):
        self.fastvgs()

    def fastvgs_plus_coco(self):
        self.fastvgs()

    def mms(self):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Model
        ckpt_pth = self.ckpt_pth.replace("mms_1b_all", "mms_1b")
        encoder, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_pth])
        cache_dir = "/share/data/speech/hackathon_2022/pretrained_models/mms_models/"
        if self.model_type == "finetuned":
            model_id = "facebook/mms-1b-all"
            target_lang = "eng"
            model = Wav2Vec2ForCTC.from_pretrained(model_id, target_lang=target_lang, ignore_mismatched_sizes=True, cache_dir=cache_dir)
            encoder = model._modules['wav2vec2']
        else:
            model_id = "facebook/mms-1b"
            encoder = Wav2Vec2Model.from_pretrained(model_id, cache_dir=cache_dir)
        return encoder, task.cfg

    def vghubert(self):
        sys.path.append(self.pckg_dir)
        from models import audio_encoder

        audio_branch_keys = ['audio_encoder', 'audio_cls_token_proj']
        other_keys = ['visual_cls_token_proj', 'trm']
        weights = torch.load(f"{self.ckpt_pth}/best_bundle.pth")
        # self.cnt_num_params(weights, audio_branch_keys, other_keys, main_key='dual_encoder')
        
        args = load_dct(f"{self.ckpt_pth}/args.pkl")
        model = audio_encoder.AudioEncoder(args)
        model.carefully_load_state_dict(weights['dual_encoder'], load_all=True)
        return model, args

    def randominit(self):
        """
        Uses pre-trained model ckpt to obtain model arguments and task config
        """
        import fairseq.models.wav2vec.wav2vec2 as w2v
        _, args, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.ckpt_pth])
        task_cfg = task.cfg
        cfg_cls = w2v.Wav2Vec2Config(**args['model'])
        encoder = w2v.Wav2Vec2Model(cfg_cls)
        return encoder, task_cfg


class DataLoader:
    def __init__(
        self,
        wav_fn,
        task_cfg=None,
    ):
        self.audio, self.fs = sf.read(wav_fn)
        if self.fs != 16000:
            import librosa
            self.audio, self.fs = librosa.load(wav_fn, sr=16000)
        self.task_cfg = task_cfg

    def stacker(self, feats, stack_order):
        """
        Concatenating consecutive audio frames
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % stack_order != 0:
            res = stack_order - len(feats) % stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, stack_order, feat_dim)).reshape(
            -1, stack_order * feat_dim
        )
        return feats

    def wavembed(self):
        from transformers import Wav2Vec2CTCTokenizer,Wav2Vec2FeatureExtractor,Wav2Vec2Processor
        speech_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('facebook/wav2vec2-base-960h')
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=speech_tokenizer)
        in_data = processor(self.audio, sampling_rate=16000,return_tensors='pt').input_values
        return in_data
    
    def shubert(self):
        from transformers import Wav2Vec2CTCTokenizer,Wav2Vec2FeatureExtractor,Wav2Vec2Processor
        speech_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('facebook/wav2vec2-base-960h')
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=speech_tokenizer)
        in_data = processor(self.audio, sampling_rate=16000,return_tensors='pt').input_values
        return in_data

    def avhubert(self):
        # 26-dim logfbank as input features
        from python_speech_features import logfbank

        audio_feats = logfbank(self.audio, samplerate=self.fs).astype(
            np.float32
        )  # [T, F]
        in_data = self.stacker(audio_feats, self.task_cfg.stack_order_audio)
        # [T/stack_order_audio, stack_order_audio*F]
        in_data = torch.from_numpy(in_data.astype(np.float32))
        if self.task_cfg.normalize:
            with torch.no_grad():
                in_data = F.layer_norm(in_data, in_data.shape[1:])
        in_data = torch.unsqueeze(in_data, 0)
        return in_data  # BxTxF

    def wavlm(self):
        in_data = torch.from_numpy(np.expand_dims(self.audio, 0).astype("float32"))
        if self.task_cfg.normalize:
            in_data = F.layer_norm(in_data, in_data.shape)
        return in_data

    def fairseq_indata(self):
        in_data = torch.from_numpy(np.expand_dims(self.audio, 0).astype("float32"))
        if self.task_cfg.normalize:
            in_data = F.layer_norm(in_data, in_data.shape)
        return in_data

    def wav2vec(self):
        in_data = self.fairseq_indata()
        return in_data

    def xlsr128(self):
        in_data = self.fairseq_indata()
        return in_data

    def xlsr53(self):
        in_data = self.fairseq_indata()
        return in_data

    def hubert(self):
        in_data = self.fairseq_indata()
        return in_data

    def mms(self):
        in_data = self.fairseq_indata()
        return in_data

    def data2vec(self):
        in_data = torch.from_numpy(np.expand_dims(self.audio, 0).astype("float32"))
        if self.task_cfg["normalize"]:
            in_data = F.layer_norm(in_data, in_data.shape)
        return in_data
    
    def vghubert(self):
        in_data = self.fairseq_indata()
        return in_data

    def randominit(self):
        in_data = self.fairseq_indata()
        return in_data
        
    def fastvgs(self):
        assert self.fs == 16000
        in_data = (self.audio - np.mean(self.audio)) / np.std(self.audio)
        in_data = torch.from_numpy(np.expand_dims(in_data, 0).astype("float32"))
        return in_data

    def fastvgs_coco(self):
        self.fastvgs()

    def fastvgs_plus_coco(self):
        self.fastvgs()

    def fastvgs_places(self):
        self.fastvgs()

    def s3adapt(self):
        in_data = self.fairseq_indata()
        return in_data

class FeatExtractor:
    def __init__(
        self,
        encoder,
        utt_id,
        wav_fn,
        rep_type,
        model_name,
        fbank_dir=None,
        task_cfg=None,
        offset=False,
        mean_pooling=False,
        ind_frame=False,
    ):
        data_obj = DataLoader(wav_fn, task_cfg)
        self.utt_id = utt_id
        self.task_cfg = task_cfg
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if "s3prl" in model_name:
            self.in_data = getattr(data_obj, self.model_name.split("_")[1])().to(
                self.device
            )
        elif "mms" in model_name:
            self.in_data = getattr(data_obj, "mms")().to(
                self.device
            )
        else:
            self.in_data = getattr(data_obj, self.model_name.split("_")[0])().to(
                self.device
            )
        
        self.encoder = encoder
        if "fastvgs" in model_name:
            for idx in range(2):
                self.encoder[idx].eval()
                self.encoder[idx].to(self.device)
        else:
            self.encoder.eval()
            self.encoder.to(self.device)
        self.offset = offset
        self.mean_pooling = mean_pooling
        self.ind_frame = ind_frame
        self.rep_type = rep_type
        if self.rep_type == "local":
            self.fbank_dir = fbank_dir
            self.fbank = np.load(os.path.join(fbank_dir, utt_id + ".npy"))
        
        self.contextualized_features = {}
        self.local_features = {}
        self.cls_features = {}

    def wavembed(self):
        with torch.no_grad():
            feat_vec = self.encoder(self.in_data).encoder_last_hidden_state
        self.contextualized_features[0] = feat_vec.squeeze().cpu().numpy()
        self.stride_sec = 20 / 1000 # not verified if this is true

    def s3adapt(self):
        """
        Extract representations from PET model
        """
        layer_reps = self.encoder(self.in_data)["hidden_states"]
        if self.rep_type == "contextualized":
            for layer_num, layer_rep in enumerate(layer_reps):
                self.contextualized_features[layer_num] = (
                    np.squeeze(layer_rep.detach().cpu().numpy(), axis=0)
                )
                num_frames = len(self.contextualized_features[layer_num])
                if layer_num == 0:
                    self.n_frames = num_frames
                else:
                    assert self.n_frames == num_frames
                layer_num += 1
        
    def shubert(self):
        with torch.no_grad():
            feat_vec = self.encoder(self.in_data).last_hidden_state
        self.contextualized_features[0] = feat_vec.squeeze().cpu().numpy()
        self.stride_sec = 20 / 1000 # not verified if this is true

    def avhubert(self):
        # model only has a projection layer before the transformer module
        with torch.no_grad():
            # Specify output_layer if you want to extract feature of an intermediate layer
            _, all_features, in_rep, _ = self.encoder.extract_finetune(
                source={"video": None, "audio": self.in_data.transpose(1, 2)},
                output_layer=None,
                padding_mask=None,
            )
        self.contextualized_features[0] = (
            in_rep.transpose(1, 2).squeeze(0).cpu().numpy()
        )
        layer_num = 1
        for layer_rep, _ in all_features:
            self.contextualized_features[layer_num] = layer_rep.squeeze(1).cpu().numpy()
            layer_num += 1
        self.stride_sec = 40 / 1000
        self.n_frames = len(in_rep.transpose(1, 2).squeeze(0))

    def wavlm(self):
        with torch.no_grad():
            output = self.encoder.extract_features(
                self.in_data,
                output_layer=self.encoder.cfg.encoder_layers,
                ret_layer_results=True,
                ret_conv=True,
            )
        self.attn_weights_lst = self.encoder.encoder.all_attn_weights
        rep, layer_results = output[0]
        self.local_features = output[2]
        layer_num = 0
        for layer_rep, _ in layer_results:
            self.contextualized_features[layer_num] = (
                layer_rep.transpose(0, 1).squeeze(0).cpu().numpy()
            )
            layer_num += 1
        self.n_frames = self.contextualized_features[0].shape[0]
        
        self.stride_sec = 20 / 1000

    def vghubert(self):
        with torch.no_grad():
            all_feats = self.encoder(self.in_data, padding_mask=None, mask=False, need_attention_weights=False, superb=True)
        if self.rep_type == "contextualized":
            for layer_num, layer_rep in enumerate(all_feats):
                self.contextualized_features[layer_num] = (
                    layer_rep.squeeze(0).cpu().numpy()[1:]
                )
            self.n_frames = len(self.contextualized_features[0])
        elif self.rep_type == "cls":
            for layer_num, layer_rep in enumerate(all_feats):
                self.cls_features[layer_num] = (
                    layer_rep.squeeze(0).cpu().numpy()[:1]
                )
        self.stride_sec = 20 / 1000
   
    def fairseq_extractor(self):
        with torch.no_grad():
            in_rep, local_features = self.encoder.feature_extractor(self.in_data)
            encoder_out = self.encoder.forward(self.in_data, mask=False, features_only=True)
            if self.rep_type == "quantized" and "hubert" not in self.model_name:
                self.z_discrete, self.indices = self.encoder.quantize(self.in_data)
        # self.attn_weights_lst = self.encoder.encoder.all_attn_weights
        if self.rep_type == "contextualized":
            for layer_num, layer_rep in enumerate(encoder_out["layer_results"]):
                self.contextualized_features[layer_num] = (
                    layer_rep[0].squeeze(1).cpu().numpy()
                )
        if self.rep_type == "local":
            self.local_features = local_features
        self.n_frames = len(in_rep.transpose(1, 2).squeeze(0))
        self.stride_sec = 20 / 1000

    def mms(self):
        with torch.no_grad():
            encoder_out = self.encoder(self.in_data, output_hidden_states=True)["hidden_states"]
        if self.rep_type == "contextualized":
            for layer_num, layer_rep in enumerate(encoder_out):
                self.contextualized_features[layer_num] = (
                    layer_rep.squeeze(0).cpu().numpy()
                )
            self.n_frames = len(self.contextualized_features[0])
        self.stride_sec = 20 / 1000

    def wav2vec(self):
        self.fairseq_extractor()

    def xlsr53(self):
        self.fairseq_extractor()

    def xlsr128(self):
        self.fairseq_extractor()

    def hubert(self):
        self.fairseq_extractor()

    def data2vec(self):
        self.fairseq_extractor()
        # with torch.no_grad():
        #     encoder_out = self.encoder.forward(self.in_data, mask=False, features_only=True)
        # if self.rep_type == "contextualized":
        #     for layer_num, layer_rep in enumerate(encoder_out["layer_results"]):
        #         self.contextualized_features[layer_num] = (
        #             layer_rep[0].squeeze(1).cpu().numpy()
        #         )
        # self.n_frames = len(self.contextualized_features[0])
        # self.stride_sec = 20 / 1000

    def fastvgs(self):
        if "fastvgs_plus" in self.model_name:
            num_layers = 13
        else:
            num_layers = self.task_cfg.layer_use + 2
        with torch.no_grad():
            encoder_out = self.encoder[0](
                source=self.in_data,
                padding_mask=None,
                mask=False,
                features_only=True,
                superb=True,
                tgt_layer=None,
            )
            output = self.encoder[1].forward_audio(
                self.in_data, None, test=True, return_outputs=True
            )

        if self.rep_type == "contextualized":
            for layer_num, layer_rep in enumerate(encoder_out["hidden_states"]):
                if layer_num < num_layers:
                    self.contextualized_features[layer_num] = (
                        layer_rep.squeeze(0).cpu().numpy()  # TxD
                    )
            curr_layer_num = layer_num + 1
            for idx, layer_num in enumerate(range(curr_layer_num, curr_layer_num + 2)):
                self.contextualized_features[layer_num] = (
                    output[-1]["trm2"][idx].squeeze(0).cpu().numpy()  # TxD
                )
            self.n_frames = self.contextualized_features[0].shape[0]
        self.cls_token = output[1].squeeze(0).cpu().numpy()
        # layer_num += 1
        self.stride_sec = 20 / 1000
        self.stride_sec_downsampled = (
            16 * self.stride_sec
        )  # resdavenet downsamples it by a factor of 16

    def randominit(self):
        self.fairseq_extractor()
        
    def transform_rep(self, kernel_size, stride, layer_rep):
        """
        Transform local z representations to match the fbank features' stride and receptive field
        layer_rep: torch.cuda.FloatTensor # B*C*T
        """
        layer_rep = torch.transpose(layer_rep, 1, 0)  # 512 x 1 x num_frames
        weight = (
            torch.from_numpy(np.ones([1, 1, kernel_size]) / kernel_size)
            .type(torch.cuda.FloatTensor)
            .to(self.device)
        )
        transformed_rep = F.conv1d(layer_rep, weight, stride=stride)
        transformed_rep = torch.transpose(transformed_rep, 1, 0)

        # check averaging
        mean_vec1 = torch.mean(layer_rep[:, :, :kernel_size], axis=-1)
        mean_vec2 = torch.mean(layer_rep[:, :, stride : stride + kernel_size], axis=-1)
        out_vec1 = transformed_rep[:, :, 0]
        out_vec2 = transformed_rep[:, :, 1]
        assert torch.mean(mean_vec1 - out_vec1) < 2e-8
        assert torch.mean(mean_vec2 - out_vec2) < 2e-8
        return torch.transpose(transformed_rep, 1, 2).squeeze(0).cpu().numpy()

    def extract_local_rep(self, rep_dct, transformed_fbank_lst, truncated_fbank_lst):
        for layer_num in range(1, len(self.local_features) + 1):
            _ = rep_dct.setdefault(layer_num, [])
            if layer_num == len(self.local_features):
                curr_layer_rep = (
                    self.local_features[layer_num - 1]
                    .transpose(1, 2)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
                rep_dct[layer_num].append(curr_layer_rep)
                num_samples_last = self.local_features[layer_num - 1].shape[-1]
            else:
                transformed_rep = self.transform_rep(
                    PER_LAYER_TRANSFORM_DCT[layer_num]["kernel"],
                    PER_LAYER_TRANSFORM_DCT[layer_num]["stride"],
                    self.local_features[layer_num - 1],
                )
                rep_dct[layer_num].append(transformed_rep)
                num_samples_rest = transformed_rep.shape[0]
        fbank = (
            torch.from_numpy(self.fbank)
            .type(torch.cuda.FloatTensor)
            .to(self.device)
            .unsqueeze(0)
        )
        if "avhubert" in self.model_name:
            kernel, stride = 1, 4
            num_samples_last = self.contextualized_features[0].shape[0]
        else:
            truncated_fbank_lst.append(self.fbank.T[:num_samples_rest])
            assert num_samples_rest < (self.fbank.shape[1] + 1)
            kernel = PER_LAYER_TRANSFORM_DCT[len(self.local_features)]["kernel"]
            stride = PER_LAYER_TRANSFORM_DCT[len(self.local_features)]["stride"]
        transformed_fbank = self.transform_rep(kernel, stride, fbank)
        assert num_samples_last < (transformed_fbank.shape[0] + 1)
        transformed_fbank_lst.append(transformed_fbank[:num_samples_last])

    def update_dct(self, indices, rep_array, rep_dct, key, label_lst, token, all_wrd_indices):
        _ = rep_dct.setdefault(key, [])
        rep_array_masked = rep_array[indices]
        if self.ind_frame:
            idx = len(rep_array_masked) // 2
            rep_array_masked = np.expand_dims(rep_array_masked[idx], 0)
        elif self.mean_pooling:
            rep_array_masked = np.expand_dims(np.mean(rep_array_masked, 0), 0)
        if key == 0 and label_lst is not None:
            start_idx = len(label_lst)
            label_lst.extend([token]*len(rep_array_masked))
            end_idx = len(label_lst)
            all_wrd_indices.append((start_idx, end_idx))
        rep_dct[key].append(rep_array_masked)

    def process_attn_weights(self, ratio_dct, time_stamp_lst, thresh=0.5):
        """
        Process the self attention weights to find the ratio of information contained within a segment
        thresh: upper threshold for entropy
        """
        from scipy.stats import entropy
        for layer_idx, attn_wt_tensor in enumerate(self.attn_weights_lst):
            _ = ratio_dct.setdefault(layer_idx, {})
            tot_ratio = 0
            # atth_weights: num_heads x target x source
            attn_weights = attn_wt_tensor.squeeze(0).cpu().numpy()
            num_heads = attn_weights.shape[0]
            num_frames = attn_weights.shape[-1]
            max_entropy = entropy(np.ones(num_frames)/num_frames)
            assert attn_weights.shape[-2] == num_frames
            for start_time, end_time, _ in time_stamp_lst:
                indices = self.get_segment_idx(
                    start_time, end_time, num_frames, self.stride_sec
                )
                exclude_indices = np.array(list(set(list(np.arange(num_frames))) - set(list(indices))))
                wt_to_target_indices = attn_weights[:,indices]
                normalized_attn_entropy = np.round(entropy(wt_to_target_indices, axis=-1)/max_entropy, 2)
                mask_mat = normalized_attn_entropy < thresh
                wt_inside = wt_to_target_indices[:, :, indices]
                wt_outside = wt_to_target_indices[:, :, exclude_indices]
                # tot_wt_inside = np.sum(np.sum(wt_inside, -1), -1) # n_heads x n_indices --> n_heads
                # tot_wt_outside = np.sum(np.sum(wt_outside, -1), -1) # n_heads x n_indices --> n_heads
                # ratio_per_head = np.round(100*tot_wt_inside/(tot_wt_outside+tot_wt_inside))
                if np.sum(mask_mat) == 0:
                    continue
                tot_wt_inside = mask_mat*np.sum(wt_inside, -1) # n_heads x n_indices 
                tot_wt_outside = mask_mat*np.sum(wt_outside, -1) # n_heads x n_indices 
                try:
                    assert np.round(np.sum(tot_wt_outside+tot_wt_inside), 1) == np.round(np.sum(mask_mat), 1)
                except:
                    print(np.sum(tot_wt_outside+tot_wt_inside))
                    print(np.sum(mask_mat))
                for head_idx in range(num_heads):
                    if np.sum(mask_mat[head_idx]) == 0:
                        continue
                    _ = ratio_dct[layer_idx].setdefault(head_idx, [])
                    fraction = np.sum(tot_wt_inside[head_idx])/np.sum(mask_mat[head_idx])
                    ratio_dct[layer_idx][head_idx].append(fraction)

    def get_segment_idx(self, start_time, end_time, len_utt, stride_sec):
        start_id = int(np.floor(float(start_time) / stride_sec))
        end_id = int(np.ceil(float(end_time) / stride_sec))
        if self.offset:
            offset = int(np.floor((end_id - start_id + 1) / 4))
            # offset = int(np.floor((end_id - start_id + 1) / 3))
            start_id += offset
            end_id -= offset
        if end_id == start_id:
            end_id += 1
        if end_id == len_utt + 1:
            end_id = len_utt
        assert end_id > start_id

        return np.arange(start_id, end_id)

    def extract_contextualized_rep(self, rep_dct, time_stamp_lst=None, label_lst=None, num_frames=None, all_wrd_indices=None):
        if self.model_name == "fastvgs_coco":
            num_layers = 9
        else:
            num_layers = len(self.contextualized_features)
        for layer_num in range(num_layers):
            c_rep = self.contextualized_features[layer_num]
            if time_stamp_lst:
                if "fastvgs" not in self.model_name or layer_num < 13:
                    stride_sec = self.stride_sec
                else:
                    stride_sec = self.stride_sec_downsampled
                for start_time, end_time, token in time_stamp_lst:
                    indices = self.get_segment_idx(
                        start_time, end_time, len(c_rep), stride_sec
                    )
                    self.update_dct(indices, c_rep, rep_dct, layer_num, label_lst, token, all_wrd_indices)
                if layer_num == 0 and num_frames is not None:
                    num_frames.append(len(indices))
            else:
                self.update_dct(np.arange(0, len(c_rep)), c_rep, rep_dct, layer_num, None, None, None)
                if layer_num == 0 and num_frames is not None:
                    num_frames.append(len(c_rep))
                # self.update_dct(np.arange(0, self.n_frames), c_rep, rep_dct, layer_num)
    
    def extract_cls_rep(self, rep_dct):
        num_layers = len(self.cls_features)
        for layer_num in range(num_layers):
            _ = rep_dct.setdefault(layer_num, [])
            rep_dct[layer_num].append(self.cls_features[layer_num])

    def extract_quantized_rep(
        self,
        quantized_features,
        quantized_indices,
        quantized_features_dct,
        discrete_indices_dct,
    ):
        idx_lst = np.arange(0, self.n_frames)
        z_discrete = self.z_discrete.squeeze(0).cpu().numpy()[idx_lst]
        indices = self.indices.squeeze(0).cpu().numpy()[idx_lst]
        quantized_features.append(z_discrete)
        quantized_indices.append(indices)
        assert self.utt_id not in quantized_features_dct
        quantized_features_dct[self.utt_id] = z_discrete
        discrete_indices_dct[self.utt_id] = indices

    def save_rep_to_file(self, rep_dct, out_dir, num_frames=None):
        for layer_num, rep_lst in rep_dct.items():
            rep_mat = np.concatenate(rep_lst, 0)
            out_fn = os.path.join(out_dir, "layer_" + str(layer_num) + ".npy")
            np.save(out_fn, rep_mat)

        if num_frames is not None:
            out_fn = os.path.join(out_dir, "num_frames.npy")
            np.save(out_fn, num_frames)

    # def extract_cls_rep(self, cls_features):
    #     cls_features.append(self.cls_token)
