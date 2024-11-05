import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPTextModel, AutoTokenizer, CLIPTextModelWithProjection


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, **kwargs):
        super().__init__()

        self.is_loaded = False

        object.__setattr__(self, 'llm_pointer', kwargs.get("llm_pointer", None))
        object.__setattr__(self, 'llm_tokenizer', kwargs.get("llm_tokenizer", None))

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.mm_vision_token_compression_type = getattr(args, 'mm_vision_token_compression_type', None)
        self.mm_vision_output_combined_token_count = getattr(args, 'mm_vision_output_combined_token_count', None)
        self.nlp = None


        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        if 'query' in self.mm_vision_token_compression_type:
            if self.mm_vision_token_compression_type == "query-embed-local-conv-self-attn-deep":
                self.text_tower = CLIPTextModelWithProjection.from_pretrained(self.vision_tower_name, device_map=device_map)
            else:
                self.text_tower = CLIPTextModel.from_pretrained(self.vision_tower_name, device_map=device_map)
            self.text_tower.requires_grad_(False)
            self.clip_tokenizer = AutoTokenizer.from_pretrained(self.vision_tower_name)

        self.is_loaded = True
    
    #adapting feature select for extra layers in vision encoder and adding relevant if/else for backward compatibility
    def feature_select(self, image_forward_outs, layers=[12,16,22,23]):
        image_feature_list = []
        for l in layers:
            image_feature_list.append(image_forward_outs.hidden_states[l])
        image_features_multi = torch.cat(image_feature_list, dim=2)
        image_features = image_forward_outs.hidden_states[self.select_layer]

        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
            image_features_multi = image_features_multi[:, 1:]

        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        
        if self.mm_vision_token_compression_type in ['quecc']:
            return image_features, image_features_multi
        return image_features
    
    @torch.no_grad()
    def forward(self, images, text):
        if type(images) is list:
            if self.mm_vision_token_compression_type in ['quecc']:
                raise NotImplementedError('The QueCC compression type is not supported for lists of images for now.')

            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            
            if self.mm_vision_token_compression_type in ['quecc']:
                image_features, image_features_multi = self.feature_select(image_forward_outs)
                image_features = image_features.to(images.dtype)
                image_features_multi = image_features_multi.to(images.dtype)
                
                #text
                text_input_tokens = self.llm_tokenizer(text, padding=True, return_tensors='pt').to(device=self.device)
                last_one_positions = torch.argmin(text_input_tokens['attention_mask'], axis=1) - 1
                output = super(self.llm_pointer.__class__, self.llm_pointer).forward(**text_input_tokens, output_hidden_states=True)
        
                text_features = output.hidden_states[-1][torch.arange(image_features.size()[0]), last_one_positions, :] #text feature for last token at last layer, accounting for padding
                text_features = text_features.unsqueeze(1).to(images.dtype)
                return (image_features_multi, text_features, image_features)
            else:
                image_features = self.feature_select(image_forward_outs).to(images.dtype)
        
        return image_features


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
