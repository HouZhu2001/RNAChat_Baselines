from xml.parsers.expat import model
import torch
from torch import nn
from rnachat.common.registry import registry
from rnachat.models.blip2 import Blip2Base, disabled_train
from rinalmo.pretrained import get_pretrained_model
from rnachat.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer,BitsAndBytesConfig
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel
from argparse import ArgumentParser
from typing import List
from rnachat.params_eval import analyze_all_checkpoints, count_changed_layers, count_changed_params, list_detailed_param_comparison, save_comprehensive_loading_report
import copy
@registry.register_model("rnachat_abl6")

# def get_device_map() -> str:
#     return 'cuda' if torch.cuda.is_available() else 'cpu'

# device = get_device_map()  # 'cpu'
class RNAChatAbl6(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_llama": "",
    }
    
    def __init__(self,
                 device=torch.device("cpu"),
                 freeze_rna_encoder=True,
                 llama_model="",
                 freeze_llama=True,
                 freeze_lp=False,
                 max_txt_len=32,
                 low_resource=False,  # use 8 bit and put vit in cpu
                 end_sym='\n',):
        super().__init__()
        print("RNAChatAbl6")
        print('Loading RNA encoder')
        self.rna_encoder, self.alphabet = get_pretrained_model(model_name="giga-v1")
        print("Loaded RiNALMo encoder")
        if freeze_rna_encoder:
            for name, param in self.rna_encoder.named_parameters():
                param.requires_grad = False
            self.rna_encoder = self.rna_encoder.eval()
            self.rna_encoder.train = disabled_train
            # logging.info("freeze rna encoder")
        else:
            self.rna_encoder = self.rna_encoder.train()
            
        parser = ArgumentParser()
        self.args_ = parser.parse_args()
        self.args_.device = torch.cuda.current_device()
        self.low_resource = low_resource
        
        self.tokenizer = self.alphabet.batch_tokenize
        print('Loading LLAMA model')
        print("Llama model: ", llama_model)
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)

        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
        if self.low_resource:
            print("Start Low Resource Mode")
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=torch.bfloat16
            #     )
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map='auto',
                # quantization_config=bnb_config
                # load_in_8bit_fp32_cpu_offload=True,
                # device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                device_map='auto',
            )

        # Compatibility shim for code paths that expect PEFT-like nested attributes
        # Provide minimal, non-recursive structure:
        #   - llama_model.model.model -> llama_model (terminal)
        #   - llama_model.base_model.model -> llama_model (terminal)
        try:
            import types
            llama = self.llama_model
            # Ensure llama.model exists and has a terminal .model that points back to llama
            if not hasattr(llama, 'model') or not hasattr(llama.model, 'model'):
                ns = types.SimpleNamespace()
                ns.model = llama  # terminal
                setattr(llama, 'model', ns)
            # Ensure llama.base_model.model exists and is terminal as well
            if not hasattr(llama, 'base_model') or not hasattr(getattr(llama, 'base_model'), 'model'):
                base_ns = types.SimpleNamespace()
                base_ns.model = llama  # terminal
                setattr(llama, 'base_model', types.SimpleNamespace(model=llama))
        except Exception:
            pass

        if freeze_llama:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        else:
            lora_target_modules: List[str] = ["q_proj", "v_proj"]
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llama_model = get_peft_model(self.llama_model, config)
            self.llama_model.print_trainable_parameters()

        self.rinalmo_llama_proj = nn.Linear(
            1280, self.llama_model.config.hidden_size
        )

        if freeze_lp:
            for name, param in self.rinalmo_llama_proj.named_parameters():
                param.requires_grad = False
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
       

    def encode_rna(self, rna):
        llama_device = next(self.llama_model.parameters()).device
        inputs = torch.tensor(self.alphabet.batch_tokenize(rna), dtype=torch.int64, device=llama_device)
        # Allow gradients to flow through RiNALMo when it's unfrozen; otherwise, no_grad
        rna_requires_grad = any(p.requires_grad for p in self.rna_encoder.parameters())
        grad_ctx = torch.enable_grad() if rna_requires_grad else torch.no_grad()
        with grad_ctx:
            with torch.cuda.amp.autocast():
                outputs = self.rna_encoder(inputs)['representation']
        if outputs.dtype != self.rinalmo_llama_proj.weight.dtype:
            outputs = outputs.to(self.rinalmo_llama_proj.weight.dtype)

        inputs_llama = self.rinalmo_llama_proj(outputs)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(outputs.device)
        return inputs_llama, atts_llama

    def prompt_list_wrap(self, rna_embeds, atts_rna, prompt):
        if prompt:
            p_before_lst = []
            p_after_lst = []
            for p in prompt:
                p_before, p_after = p.split('<RNAHere>')
                p_before_lst.append(p_before)
                p_after_lst.append(p_after)
            p_before_tokens_lst = self.llama_tokenizer(
                p_before_lst, return_tensors="pt", add_special_tokens=False).to(rna_embeds.device)

            p_after_tokens_lst = self.llama_tokenizer(
                p_after_lst, return_tensors="pt", add_special_tokens=True, padding=True).to(rna_embeds.device)
            
            p_before_embeds = self.llama_model.get_input_embeddings()(p_before_tokens_lst.input_ids)
            p_after_embeds = self.llama_model.get_input_embeddings()(p_after_tokens_lst.input_ids)
            wrapped_rna_embeds = torch.cat([p_before_embeds, rna_embeds, p_after_embeds], dim=1)
            wrapped_atts_rna = atts_rna[:, :1].expand(-1, wrapped_rna_embeds.shape[1])
            return wrapped_rna_embeds, wrapped_atts_rna
        else:
            return rna_embeds, atts_rna

    def forward(self, samples):
        seqs = samples["seq"] # list of seq
        # print(samples)
        rna_embeds, atts = self.encode_rna(seqs)

        rna_embeds, atts_rna = self.prompt_list_wrap(rna_embeds, atts, samples["prompt"])

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(rna_embeds.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_rna.shape[0], atts_rna.shape[1]+1],
                       dtype=torch.long).to(rna_embeds.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = rna_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.get_input_embeddings()(bos)
        atts_bos = atts_rna[:, :1]

        to_regress_embeds = self.llama_model.get_input_embeddings()(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, rna_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_rna, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        logits = outputs.logits
        logits = torch.argmax(logits, dim=2)
        loss = outputs.loss
        return {"loss": loss}
    
    @classmethod
    def from_config(cls, cfg):

        llama_model = cfg.get("llama_model")
        print("Llama model from cfg: ", llama_model)

        freeze_rna_encoder = cfg.get("freeze_rna_encoder", False)
        freeze_lp = cfg.get("freeze_lp", False)
        freeze_llama = cfg.get("freeze_llama", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        embedding_agg = cfg.get("embedding_agg", 1)

        model = cls(
            device= device_8bit,
            freeze_rna_encoder=freeze_rna_encoder,
            freeze_lp=freeze_lp,
            freeze_llama=freeze_llama,
            llama_model=llama_model,
            # embedding_agg = embedding_agg, 
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            # device_8bit=device_8bit,
        )

        original_state = copy.deepcopy(model.state_dict())

        # results = analyze_all_checkpoints(
        #     model=model,
        #     checkpoint_paths={
        #         'stage1': 'rnachat/checkpoints/checkpoint_stage1.pth',
        #         'peft': 'rnachat/checkpoints/checkpoint_stage2.pth'
        #     }
        # )

        # results = list_detailed_param_comparison(
        #     model=model,
        #     checkpoint_paths={
        #         'STAGE1': 'rnachat/checkpoints/checkpoint_stage1.pth',
        #         'PEFT': 'rnachat/checkpoints/checkpoint_stage2.pth'
        #     }
        # )

        # model_dict = model.state_dict()
        # msg1, msg2 = None, None

        # stage1_ckpt = cfg.get("stage1_ckpt", "")  # load weights of encoder and LP
        # if stage1_ckpt:
        #     import os
        #     print(os.getcwd())
        #     print("Load Checkpoint1: {}".format(stage1_ckpt))
        #     assert os.path.exists(stage1_ckpt), "Checkpoint file does not exist!"
        #     ckpt = torch.load(stage1_ckpt, map_location="cpu")
        #     assert "model" in ckpt, "[ERROR] 'model' key not found in checkpoint!"
            
        #     msg = model.load_state_dict(ckpt['model'], strict=False)
        #     stage1_dict = ckpt['model'] if 'model' in ckpt else ckpt
        #     # Filter loadable parameters from STAGE1
        #     stage1_loadable = {}
        #     for key, value in stage1_dict.items():
        #         if key in model_dict and model_dict[key].shape == value.shape:
        #             stage1_loadable[key] = value
        #     # Load STAGE1 parameters
        #     msg1 = model.load_state_dict(stage1_dict, strict=False)
            
        #     print(f"✅ STAGE1 loaded successfully!")
        #     print(f"   - Available in checkpoint: {len(stage1_dict)}")
        #     print(f"   - Loadable Keys: {len(stage1_loadable)}")
        #     print(f"   - Missing keys: {len(msg1.missing_keys)}")
        #     print(f"   - Unexpected keys: {len(msg1.unexpected_keys)}")

    
        # # Check for meta tensors
        # has_meta = False
        # for name, param in model.named_parameters():
        #     if param.device.type == "meta":
        #         print(f"[ERROR] Parameter still on meta: {name}")
        #         has_meta = True

        # if has_meta:
        #     raise RuntimeError("Model still has meta tensors after loading checkpoint!")
        
        # # Step 2: Load PEFT (LoRA weights)
        # print(f"\n🔄 Step 2: Loading PEFT checkpoint (LoRA weights)")
        # print("-" * 60)

        # peft_path = cfg.get("peft_ckpt", "")  # load weights of LoRA
        # if peft_path:
        #     peft_ckpt = torch.load(peft_path, map_location='cpu')
        #     peft_dict = peft_ckpt['model'] if 'model' in peft_ckpt else peft_ckpt

            
        #     # Filter loadable parameters from PEFT
        #     peft_loadable = {}
        #     for key, value in peft_dict.items():
        #         if key in model_dict and model_dict[key].shape == value.shape:
        #             peft_loadable[key] = value
            
        #     # Load PEFT parameters
        #     msg2 = model.load_state_dict(peft_dict, strict=False)
            
        #     print(f"✅ PEFT loaded successfully!")
        #     print(f"   - Available in checkpoint: {len(peft_dict)}")
        #     print(f"   - Loadable Keys: {len(peft_loadable)}")
        #     print(f"   - Missing keys: {len(msg2.missing_keys)}")
        #     print(f"   - Unexpected keys: {len(msg2.unexpected_keys)}")
            
        # save_comprehensive_loading_report(msg1, msg2, model, "rnachat/checkpoints/checkpoint_stage1.pth", "rnachat/checkpoints/checkpoint_stage2.pth")

        stage1_ckpt = cfg.get("stage1_ckpt", "")
        peft_ckpt = cfg.get("peft_ckpt", "")
        
        print("\n🎯 CORRECT TWO-STAGE LORA LOADING")
        print("=" * 60)
        
        # Step 1: Load RNA encoder + projection layer from Stage 1
        # CRITICAL: Skip ALL llama_model weights!
        import os
        if stage1_ckpt and os.path.exists(stage1_ckpt):
            print("📦 Loading RNA encoder + projection from Stage 1...")
            stage1_checkpoint = torch.load(stage1_ckpt, map_location="cpu")
            stage1_dict = stage1_checkpoint['model'] if 'model' in stage1_checkpoint else stage1_checkpoint
            
            # Define what to load from Stage 1
            allowed_prefixes = [
                'rna_encoder.',           # RNA encoder weights  
                'rinalmo_llama_proj.',    # Projection layer weights
            ]
            
            # Define what to NEVER load (these break LoRA!)
            forbidden_prefixes = [
                'llama_model.',           # ALL LLaMA weights forbidden!
            ]
            
            # Filter Stage 1 weights
            stage1_filtered = {}
            skipped_llama = 0
            loaded_categories = {'rna_encoder': 0, 'projection': 0, 'other': 0}
            
            for key, value in stage1_dict.items():
                # Skip forbidden keys (LLaMA weights)
                if any(key.startswith(forbidden) for forbidden in forbidden_prefixes):
                    skipped_llama += 1
                    continue
                
                # Load allowed keys
                should_load = any(key.startswith(allowed) for allowed in allowed_prefixes)
                
                if should_load and key in model.state_dict():
                    if model.state_dict()[key].shape == value.shape:
                        stage1_filtered[key] = value
                        
                        # Categorize for reporting
                        if 'rna_encoder' in key:
                            loaded_categories['rna_encoder'] += 1
                        elif 'rinalmo_llama_proj' in key:
                            loaded_categories['projection'] += 1
                        else:
                            loaded_categories['other'] += 1
                    else:
                        print(f"⚠️  Shape mismatch: {key}")
            
            # Load filtered Stage 1 weights
            if stage1_filtered:
                msg1 = model.load_state_dict(stage1_filtered, strict=False)
                print(f"✅ Stage 1 loaded successfully:")
                print(f"   - RNA encoder weights: {loaded_categories['rna_encoder']}")
                print(f"   - Projection weights: {loaded_categories['projection']}")
                print(f"   - Other weights: {loaded_categories['other']}")
                print(f"   - Skipped LLaMA weights: {skipped_llama} (GOOD!)")
                
                # These should be minimal since we're loading selectively
                if msg1.unexpected_keys:
                    print(f"   - Unexpected keys: {len(msg1.unexpected_keys)}")
            else:
                print("❌ No valid weights found in Stage 1")
                return None
        
        # Step 2: Load LoRA weights from Stage 2
        if peft_ckpt and os.path.exists(peft_ckpt):
            print("\n🎯 Loading LoRA weights from Stage 2...")
            peft_checkpoint = torch.load(peft_ckpt, map_location='cpu')
            peft_dict = peft_checkpoint['model'] if 'model' in peft_checkpoint else peft_checkpoint
            
            # Filter LoRA weights and skip buffers
            lora_weights = {}
            buffer_skipped = 0
            lora_categories = {'lora_A': 0, 'lora_B': 0, 'other': 0}
            
            # Buffer patterns to skip
            skip_buffer_patterns = ['rotary_emb.inv_freq', 'cos_cached', 'sin_cached']
            
            for key, value in peft_dict.items():
                # Skip buffer conflicts
                if any(pattern in key for pattern in skip_buffer_patterns):
                    buffer_skipped += 1
                    continue
                
                # Load compatible weights
                if key in model.state_dict() and model.state_dict()[key].shape == value.shape:
                    lora_weights[key] = value
                    
                    # Categorize
                    if 'lora_A' in key:
                        lora_categories['lora_A'] += 1
                    elif 'lora_B' in key:
                        lora_categories['lora_B'] += 1
                    else:
                        lora_categories['other'] += 1
            
            # Load LoRA weights
            if lora_weights:
                msg2 = model.load_state_dict(lora_weights, strict=False)
                print(f"✅ Stage 2 loaded successfully:")
                print(f"   - LoRA A matrices: {lora_categories['lora_A']}")
                print(f"   - LoRA B matrices: {lora_categories['lora_B']}")
                print(f"   - Other weights: {lora_categories['other']}")
                print(f"   - Skipped buffer keys: {buffer_skipped}")
                
                # Filter real unexpected keys (ignore buffer conflicts)
                real_unexpected = [k for k in msg2.unexpected_keys 
                                if not any(p in k for p in skip_buffer_patterns)]
                
                if real_unexpected:
                    print(f"   ⚠️  Real unexpected keys: {len(real_unexpected)}")
                    for key in real_unexpected[:3]:
                        print(f"      - {key}")
                
                total_lora = lora_categories['lora_A'] + lora_categories['lora_B']
                if total_lora == 0:
                    print("❌ WARNING: No LoRA weights loaded!")
                    return None
            else:
                print("❌ No valid LoRA weights found in Stage 2")
                return None
        
        # Final verification
        print("\n🔍 Final Model Verification:")
        print("-" * 40)
        
        # Check LoRA parameters
        lora_param_count = 0
        trainable_params = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if any(lora_key in name for lora_key in ['lora_A', 'lora_B']):
                    lora_param_count += param.numel()
        
        print(f"✅ Model verification:")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        print(f"   - LoRA parameters: {lora_param_count:,}")
        print(f"   - Trainable ratio: {100 * trainable_params / total_params:.3f}%")
        
        # Check that base LLaMA weights are preserved
        print(f"   - Base LLaMA weights: PRESERVED (not loaded from Stage 1)")
        print(f"   - LoRA adapters: LOADED from Stage 2")
        print(f"   - Projection layer: LOADED from Stage 1")
        
        # Device check
        device = next(model.parameters()).device
        print(f"   - Model device: {device}")
        
        # if lora_param_count == 0:
        #     print("❌ CRITICAL: No LoRA parameters found!")
        #     return None
        
        print("\n✅ Model ready for LoRA inference!")
    

        return model
    
    @torch.no_grad()
    def generate(self, samples, max_length=100, num_beams=1, temperature=1.0, do_sample=False, repetition_penalty=1.0):
        """Generate responses from RNA sequences"""
        seqs = samples["seq"]
        prompts = samples.get("prompt", [""] * len(seqs))
        
        # Encode RNA
        rna_embeds, atts_rna = self.encode_rna(seqs)
        rna_embeds, atts_rna = self.prompt_list_wrap(rna_embeds, atts_rna, prompts)
        
        batch_size = rna_embeds.shape[0]
        device = rna_embeds.device
        
        # Start with BOS token
        generated_ids = torch.full(
            [batch_size, 1], 
            self.llama_tokenizer.bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Use LLaMA's generate method properly
        bos_embeds = self.llama_model.get_input_embeddings()(generated_ids)
        inputs_embeds = torch.cat([bos_embeds, rna_embeds], dim=1)
        
        bos_atts = torch.ones([batch_size, 1], dtype=torch.long, device=device)
        attention_mask = torch.cat([bos_atts, atts_rna], dim=1)
        
        # Generate using the model
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=max_length + inputs_embeds.shape[1],
            num_beams=num_beams,
            temperature=temperature,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.llama_tokenizer.eos_token_id,
            eos_token_id=self.llama_tokenizer.eos_token_id,
        )
        
        # Extract only the generated part (after RNA embeddings)
        generated_text = []
        for i, output in enumerate(outputs):
            # Skip the input part (BOS + RNA embeddings)
            start_idx = inputs_embeds.shape[1]
            generated_tokens = output[start_idx:]
            
            text = self.llama_tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_text.append(text.strip())
        
        return generated_text
        
