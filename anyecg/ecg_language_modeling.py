import torch
from torch import nn
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from anyecg.utils import get_label_from_input_ids
from anyecg.ecgvit import ecg_vit_base, ecg_vit_large

class ECG_Language_Model(nn.Module):
    def __init__(self, unfreeze_ecg_model=False, use_lora=False):
        super(ECG_Language_Model, self).__init__()
        # loaded the ecg encoder vit base 
        self.ecg_model = self.get_ecg_encoder(unfreeze_ecg_model)
        # load the slm phi2 
        self.language_model, self.tokenizer = self.get_language_model(use_lora)
        # define the projectior ecg-> slm 
        self.projection = nn.Sequential(
            nn.Linear(self.ecg_model.dim, self.language_model.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.language_model.config.hidden_size, self.language_model.config.hidden_size)
        )

        for param in self.projection.parameters():
            nn.init.normal_(param, std=0.02)
        
        # Track device
        self._device = torch.device('cpu')
            
        # Get ECG token IDs after they've been added to the tokenizer
        self.ecg_start_id = self.tokenizer.convert_tokens_to_ids("<ecg>")

        self.ecg_end_id = self.tokenizer.convert_tokens_to_ids("</ecg>")

        self.ecg_content_id = self.tokenizer.convert_tokens_to_ids("<ecg_pad>")
        
        self.ecg_token_len = 61
        self.ecg_tokens = [self.ecg_start_id] + [self.ecg_content_id] * self.ecg_token_len + [self.ecg_end_id]
        
        self.ecg_start_token = "<ecg>"
        self.ecg_end_token = "</ecg>"
        self.ecg_content_token = "<ecg_pad>"
        
    def get_ecg_encoder(self, unfreeze_ecg_model):
        ecg_encoder = ecg_vit_base()
        if hasattr(ecg_encoder, 'fc'):
            ecg_encoder.fc = nn.Linear(ecg_encoder.fc.in_features, ecg_encoder.fc.out_features, bias=False)

        

        # pretrained_ckpt = '/mnt/sda1/xxxx/output/anyECG/vit_base_sigmoid/model_epoch9.bin'
        # ckpt_clep = torch.load(pretrained_ckpt)
        # ckpt_ecg_encoder = {}
        # for key in ckpt_clep:
        #     if 'ecg_model' in key:
        #         ckpt_ecg_encoder[key.replace('ecg_model.model.', '')] = ckpt_clep[key]
        # res = ecg_encoder.load_state_dict(ckpt_ecg_encoder, strict=False)
        # print(f'Loading ecg_encoder: {res}')

        if not unfreeze_ecg_model:
            for param in ecg_encoder.parameters():
                param.requires_grad = False
        return ecg_encoder
    
    def to(self, device):
        """Override to method to track device changes"""
        self._device = torch.device(device) if isinstance(device, str) else device
        return super().to(device)
    
    def get_language_model(self, use_lora):
        model_id = "microsoft/phi-2"
        print(f"Loading SLM: {model_id}...")

        tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Add special ECG tokens
        tokenizer.add_special_tokens({'additional_special_tokens': ["<ecg>", "</ecg>", "<ecg_pad>"]})
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Resize token embeddings after adding special tokens
        model.resize_token_embeddings(len(tokenizer))

        for param in model.parameters():
            param.requires_grad = False

        if use_lora:
            peft_config = LoraConfig(
                target_modules=["Wqkv", "out_proj", "fc1", "fc2"],
                init_lora_weights=True,
                lora_alpha=16,
                lora_dropout=0.1,
                r=8,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model.add_adapter(peft_config)
            print("Lora adapters added for phi-2 ")
        return model, tokenizer

    def chat(self, messages, temperature=0.6):
        # phi-2 is base model not support chat templates 
        try:
            tokenizer_output = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_dict=True
            ).to(self.language_model.device)
        except Exception:
            test=""
            for msg in messages:
                test+=f"{msg['role']}: {msg['content']}\n"
            test+="assistant: "
            tokenizer_output = self.tokenizer(test,return_tensors="pt",padding=True,truncation=True,max_length=512).to(self.language_model.device)

        input_ids = tokenizer_output["input_ids"]
        attention_mask = tokenizer_output["attention_mask"]

        terminators = [
            self.tokenizer.eos_token_id,
            # self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.language_model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            attention_mask=attention_mask
        )
        return self.tokenizer.batch_decode(outputs[:, input_ids.shape[-1]:], skip_special_tokens=True)

    def ecg_chat(self, ecgs, messages, temperature=0.6):
        """Generate responses for ECG and text input"""
        inputs_embeds, attention_mask, _ = self.get_input_embeds(ecgs, messages, add_generation_prompt=True)
        
        # Ensure models are on the right device
        self.language_model = self.language_model.to(self._device)
        
        # Ensure all inputs are on the right device
        inputs_embeds = inputs_embeds.to(self._device)
        attention_mask = attention_mask.to(self._device)
        
        terminators = [
            self.tokenizer.eos_token_id,
        ]
        
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=256,
            eos_token_id=terminators[0],
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            attention_mask=attention_mask
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def encode_and_project_ecg_batch(self, ecg):
        ecg_embedding = self.ecg_model.encode(ecg)
        ecg_embedding = self.projection(ecg_embedding)
        return ecg_embedding
    
    def encode_and_project_ecg(self, ecgs):
        ecg_embeddings = []
        for ecg in ecgs:
            ecg = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0).to(self._device)
            segments = []
            for i in range(0, ecg.shape[2], 1000):
                if i + 1000 <= ecg.shape[2]:
                    segments.append(ecg[:, :, i:i + 1000])
                else:
                    segment = ecg[:, :, i:]
                    padding_size = 1000 - segment.shape[2]
                    padded_segment = torch.nn.functional.pad(segment, (0, padding_size))
                    segments.append(padded_segment)
            segments = torch.cat(segments, dim=0)
            ecg_embedding = self.ecg_model.encode(segments)
            # encode and project 
            ecg_embedding = self.projection(ecg_embedding)
            # aggegate cls token and flatten pathces 
            cls_tokens = [embedding[0] for embedding in ecg_embedding]
            cls_token = torch.stack(cls_tokens).mean(dim=0)
            remaining_tokens = torch.cat([embedding[1:] for embedding in ecg_embedding], dim=0)
            ecg_embedding = torch.cat([cls_token.unsqueeze(0), remaining_tokens], dim=0)
            ecg_embeddings.append(ecg_embedding)
        return ecg_embeddings
    
    def emebed_text(self, input_ids):
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        return inputs_embeds

    def get_input_embeds(self, ecgs, messages, add_generation_prompt=False):
        debug = False
        # Ensure ecg_model is on the right device
        self.ecg_model = self.ecg_model.to(self._device)
        self.projection = self.projection.to(self._device)
        
        if isinstance(ecgs, torch.Tensor):
            ecgs = ecgs.to(self._device)
            ecg_embeddings = self.encode_and_project_ecg_batch(ecgs)
        elif isinstance(ecgs, list) and len(ecgs) > 0 and isinstance(ecgs[0], torch.Tensor):
            # List of tensors
            if len(set([ecg.shape[1] for ecg in ecgs])) == 1 and ecgs[0].shape[1] == 1000:
                ecgs = torch.stack(ecgs, dim=0).float().to(self._device)
                ecg_embeddings = self.encode_and_project_ecg_batch(ecgs)
            else:
                ecg_embeddings = self.encode_and_project_ecg(ecgs)
        else:
            ecg_embeddings = self.encode_and_project_ecg(ecgs)
        if debug:
            ecg_embeddings = [torch.zeros_like(item) for item in ecg_embeddings]
        
        # add ecg token
        for idx, message in enumerate(messages):
            ecg_token_len = ecg_embeddings[idx].shape[0]
            message[0]['content'] = self.ecg_start_token + self.ecg_content_token * ecg_token_len + self.ecg_end_token + message[0]['content']
        # get input_ids and attention_mask
        try:

            tokenizer_output = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,  # changed to 512 for mutli-ECG or long-ECG
                return_dict=True
            ).to(self.language_model.device)
        except Exception:
            test_batch=[m[0]['content'] for m in messages]
            tokenizer_output = self.tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.language_model.device)
        input_ids = tokenizer_output["input_ids"]
        attention_mask = tokenizer_output["attention_mask"]

        # get inputs_embeds
        inputs_embeds = self.emebed_text(input_ids)
        for i in range(inputs_embeds.shape[0]):
            ecg_embedding = ecg_embeddings[i]
            try:
                # Replace placeholders with real embeddings
                # Ensure dimensions match (batch, seq_len, dim)
                ecg_index = input_ids[i].tolist().index(self.ecg_start_id) + 1
                inputs_embeds[i, ecg_index:ecg_index+ecg_embedding.shape[0]] = ecg_embedding
            except ValueError:
                pass
        labels = get_label_from_input_ids(input_ids, tokenizer=self.tokenizer, device=self._device)

        # if debug:
        #     for i in range(input_ids.shape[0]):
        #         for j in range(len(input_ids[i])):
        #             print(input_ids[i][j].item(), self.tokenizer.decode(input_ids[i][j].item()), attention_mask[i][j].item(), labels[i][j].item(), inputs_embeds[i][j].tolist()[:3])
        #         print('------')
        return inputs_embeds, attention_mask, labels
    
    def forward(self, ecgs, messages):
        inputs_embeds, attention_mask, labels = self.get_input_embeds(ecgs, messages)
        outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs.loss

if __name__ == '__main__':
    ecg_language_model = ECG_Language_Model(use_lora=True).cuda()
    print("model initialized successfully with phi2")

    messages = [
        # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        
        {"role": "user", "content": "Who are you?"},
    ]
    print(ecg_language_model.chat(messages))




