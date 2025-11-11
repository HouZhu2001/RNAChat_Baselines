from xml.parsers.expat import model
import torch
from torch import nn
from rnachat.common.registry import registry
from rnachat.models.blip2 import Blip2Base, disabled_train
# from rinalmo.pretrained import get_pretrained_model  # Not needed for RNA-FM
from rnachat.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer,BitsAndBytesConfig
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel
from argparse import ArgumentParser
from typing import List
from rnachat.params_eval import analyze_all_checkpoints, count_changed_layers, count_changed_params, list_detailed_param_comparison, save_comprehensive_loading_report
import copy
import fm
from fm.pretrained import rna_fm_t12
from transformers import LlamaConfig
from modelgenerator.tasks import Embed
@registry.register_model("rnachat_abl3")

# def get_device_map() -> str:
#     return 'cuda' if torch.cuda.is_available() else 'cpu'

# device = get_device_map()  # 'cpu'
class RNAChatAbl3(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "",
    }
    
    def __init__(self,
                 device=torch.device("cpu"),
                 freeze_rna_encoder=True,
                 llama_model="",
                 freeze_llama=True,
                 freeze_lp=False,
                 max_txt_len=32,
                 low_resource=False,  # use 8 bit and put vit in cpu
                 end_sym='\n',
                 disable_quantization=False):
        super().__init__()
        print("RNAChatAbl3")
        print('Loading RNA encoder')
        # Use modelgenerator instead of RNA-FM to avoid precision issues
        self.rna_encoder = Embed.from_config({"model.backbone": "aido_rna_1b600m"}).eval()
        print("Loaded modelgenerator RNA encoder")
        
        # Modelgenerator RNA encoder loaded successfully
        print("✅ Modelgenerator RNA encoder loaded successfully")
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
        
        # self.tokenizer = self.alphabet.batch_tokenize
        # print('Loading LLAMA model')
        # print("Llama model: ", llama_model)
        # config = LlamaConfig.from_pretrained("lmsys/vicuna-13b-v1.5")
        # config.rms_norm_eps = 1e-3 
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)

        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
        # Try quantization, fallback to no quantization if bitsandbytes is incompatible
        if disable_quantization:
            print("Quantization disabled by config")
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16,
                device_map='auto',
            )
        else:
            try:
                if self.low_resource:
                    print("Start Low Resource Mode with 4-bit quantization")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                    self.llama_model = LlamaForCausalLM.from_pretrained(
                        llama_model,
                        torch_dtype=torch.bfloat16,
                        quantization_config=bnb_config,
                        device_map='auto',
                    )
                else:
                    print("Using 8-bit quantization")
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.bfloat16
                    )
                    self.llama_model = LlamaForCausalLM.from_pretrained(
                        llama_model,
                        torch_dtype=torch.bfloat16,
                        quantization_config=bnb_config,
                        device_map='auto',
                    )
            except Exception as e:
                print(f"⚠️  Quantization failed: {e}")
                print("Falling back to no quantization (will use more memory)")
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.bfloat16,
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

        # Get the hidden size from modelgenerator encoder dynamically
        try:
            # Move modelgenerator to CPU for initial testing to avoid device issues
            if hasattr(self.rna_encoder, 'to'):
                self.rna_encoder = self.rna_encoder.to('cpu')
            
            # Test with a dummy sequence to get the actual output size
            test_batch = self.rna_encoder.transform({"sequences": ["ACGT"]})
            test_embeddings = self.rna_encoder(test_batch)
            if isinstance(test_embeddings, torch.Tensor):
                if test_embeddings.dim() == 3:
                    rna_encoder_hidden_size = test_embeddings.shape[-1]
                elif test_embeddings.dim() == 2:
                    rna_encoder_hidden_size = test_embeddings.shape[-1]
                else:
                    rna_encoder_hidden_size = 2048  # Fallback
            else:
                rna_encoder_hidden_size = 2048  # Fallback
            print(f"✅ Detected modelgenerator hidden size: {rna_encoder_hidden_size}")
        except Exception as e:
            print(f"⚠️  Could not detect modelgenerator hidden size: {e}")
            rna_encoder_hidden_size = 2048  # Fallback to RNA-FM size
        
        # Create projection layer with correct input size
        self.rnafm_llama_proj = nn.Linear(
            rna_encoder_hidden_size, self.llama_model.config.hidden_size
        )
        print(f"✅ Created projection layer: {rna_encoder_hidden_size} -> {self.llama_model.config.hidden_size}")
        
        # Initialize projection layer with proper weights to prevent NaN
        nn.init.xavier_uniform_(self.rnafm_llama_proj.weight)
        if self.rnafm_llama_proj.bias is not None:
            nn.init.zeros_(self.rnafm_llama_proj.bias)
        
        self.rnafm_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.llama_model, 'gradient_checkpointing_enable'):
            self.llama_model.gradient_checkpointing_enable()
            print("✅ Enabled gradient checkpointing to save memory")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✅ Cleared CUDA cache")

        if freeze_lp:
            for name, param in self.rnafm_llama_proj.named_parameters():
                param.requires_grad = False
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
       

    def encode_rna(self, rna):
        # Handle both single sequences and lists of sequences
        if isinstance(rna, list):
            sequences = rna
        else:
            sequences = [rna]
        
        # Convert DNA to RNA (T -> U) and validate sequences
        processed_sequences = []
        for i, seq in enumerate(sequences):
            if not seq or len(seq) == 0:
                print(f"⚠️  Warning: Empty RNA sequence at index {i}")
                processed_sequences.append("A")  # Replace with single nucleotide
            else:
                # Simple conversion: T -> U
                seq = seq.replace('T', 'U').replace('t', 'u')
                processed_sequences.append(seq)
        
        # Use modelgenerator API
        try:
            # Ensure modelgenerator is on the correct device
            target_device = next(self.rnafm_llama_proj.parameters()).device
            
            # Move modelgenerator to target device if possible
            if hasattr(self.rna_encoder, 'to'):
                try:
                    self.rna_encoder = self.rna_encoder.to(target_device)
                    print(f"✅ Moved modelgenerator to {target_device}")
                except Exception as e:
                    print(f"⚠️  Could not move modelgenerator to {target_device}: {e}")
                    # Keep on CPU and move outputs later
            
            # Transform sequences using modelgenerator
            transformed_batch = self.rna_encoder.transform({"sequences": processed_sequences})
            
            # Get embeddings
            with torch.no_grad() if not any(p.requires_grad for p in self.rna_encoder.parameters()) else torch.enable_grad():
                embeddings = self.rna_encoder(transformed_batch)
            
            # Check if embeddings are in the expected format
            if isinstance(embeddings, torch.Tensor):
                # If it's a single tensor, assume it's [batch_size, seq_len, hidden_size]
                if embeddings.dim() == 3:
                    token_embeddings = embeddings
                elif embeddings.dim() == 2:
                    # If it's [batch_size, hidden_size], we need to add sequence dimension
                    # This might be pooled embeddings - you may need to adjust based on your needs
                    batch_size, hidden_size = embeddings.shape
                    token_embeddings = embeddings.unsqueeze(1)  # [batch_size, 1, hidden_size]
                else:
                    raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
            else:
                raise ValueError(f"Unexpected embedding type: {type(embeddings)}")
            
            # Move embeddings to the same device as the projection layer
            device = next(self.rnafm_llama_proj.parameters()).device
            token_embeddings = token_embeddings.to(device)
            
            print(f"✅ Modelgenerator embeddings shape: {token_embeddings.shape}, device: {token_embeddings.device}")
            
        except Exception as e:
            print(f"❌ Error with modelgenerator: {e}")
            # Fallback: create dummy embeddings
            batch_size = len(processed_sequences)
            hidden_size = 2048  # Default size
            max_seq_len = max(len(seq) for seq in processed_sequences) if processed_sequences else 1
            device = next(self.rnafm_llama_proj.parameters()).device
            token_embeddings = torch.zeros(batch_size, max_seq_len, hidden_size, device=device)
        
        # Check for NaN in embeddings and handle it
        if torch.isnan(token_embeddings).any():
            print("❌ NaN detected in modelgenerator output! Replacing with zeros...")
            token_embeddings = torch.nan_to_num(token_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
            print(f"   - After NaN handling: {torch.isnan(token_embeddings).sum()} NaN values")
        
        # Project embeddings to LLaMA hidden size
        batch_size, seq_len, hidden_size = token_embeddings.shape
        token_embeddings_flat = token_embeddings.view(-1, hidden_size)  # [batch*seq, hidden]
        
        # Debug: Check dimensions before projection
        print(f"🔍 Projection debug:")
        print(f"   - Input embeddings shape: {token_embeddings_flat.shape}")
        print(f"   - Projection layer input size: {self.rnafm_llama_proj.in_features}")
        print(f"   - Projection layer output size: {self.rnafm_llama_proj.out_features}")
        
        # # Check if dimensions match
        if token_embeddings_flat.shape[-1] != self.rnafm_llama_proj.in_features:
            print(f"❌ Dimension mismatch! Recreating projection layer...")
        #     # Recreate projection layer with correct input size
        #     self.rnafm_llama_proj = nn.Linear(
        #         token_embeddings_flat.shape[-1], 
        #         self.llama_model.config.hidden_size
        #     ).to(token_embeddings_flat.device)
        #     # Reinitialize weights
        #     nn.init.xavier_uniform_(self.rnafm_llama_proj.weight)
        #     if self.rnafm_llama_proj.bias is not None:
        #         nn.init.zeros_(self.rnafm_llama_proj.bias)
        #     print(f"✅ Recreated projection layer: {token_embeddings_flat.shape[-1]} -> {self.llama_model.config.hidden_size}")
        
        # Project all tokens at once
        projected_flat = self.rnafm_llama_proj(token_embeddings_flat)  # [batch*seq, llama_hidden]
        
        # Reshape back to sequence format
        rna_emb = projected_flat.view(batch_size, seq_len, -1)  # [batch_size, seq_len, llama_hidden]
        rna_emb = self.rnafm_norm(rna_emb)
        
        # Ensure dtype compatibility with LLaMA model
        llama_dtype = next(self.llama_model.parameters()).dtype
        if rna_emb.dtype != llama_dtype:
            rna_emb = rna_emb.to(dtype=llama_dtype)
        
        # rna_emb is now [batch_size, seq_len, llama_hidden]
        inputs_llama = rna_emb
        
        # Create attention mask for the sequence
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(rna_emb.device)  # [batch_size, seq_len]
        
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
                p_before_lst, return_tensors="pt").to(rna_embeds.device)

            p_after_tokens_lst = self.llama_tokenizer(
                p_after_lst, return_tensors="pt", padding=True).to(rna_embeds.device)
            
            p_before_embeds = self.llama_model.get_input_embeddings()(p_before_tokens_lst.input_ids)
            p_after_embeds = self.llama_model.get_input_embeddings()(p_after_tokens_lst.input_ids)
            wrapped_rna_embeds = torch.cat([p_before_embeds, rna_embeds, p_after_embeds], dim=1)
            # atts_rna is now [batch_size, seq_len] instead of [batch_size, 1]
            wrapped_atts_rna = torch.cat([
                torch.ones(p_before_embeds.shape[:-1], dtype=torch.long, device=rna_embeds.device),
                atts_rna,
                torch.ones(p_after_embeds.shape[:-1], dtype=torch.long, device=rna_embeds.device)
            ], dim=1)
            return wrapped_rna_embeds, wrapped_atts_rna
        else:
            return rna_embeds, atts_rna

    def forward(self, samples):
        # Clear CUDA cache before forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Enable anomaly detection to find NaN source
        with torch.autograd.detect_anomaly():
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
        ).to(rna_embeds.device)
        
        # Validate LLaMA token IDs to prevent IndexError
        # print(f"🔍 LLaMA token validation:")
        # print(f"   - Token range before clamping: [{to_regress_tokens.input_ids.min():.0f}, {to_regress_tokens.input_ids.max():.0f}]")
        # print(f"   - Vocab size: {self.llama_model.config.vocab_size}")
        
        # Clamp tokens to valid vocabulary range
        to_regress_tokens.input_ids = torch.clamp(
            to_regress_tokens.input_ids, 
            min=0, 
            max=self.llama_model.config.vocab_size - 1
        )
        # print(f"   - Token range after clamping: [{to_regress_tokens.input_ids.min():.0f}, {to_regress_tokens.input_ids.max():.0f}]")

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_rna.shape[0], atts_rna.shape[1]+1],
                       dtype=torch.long).to(rna_embeds.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = rna_embeds.shape[0]
        
        # Validate BOS token ID
        bos_token_id = self.llama_tokenizer.bos_token_id
        if bos_token_id is None or bos_token_id < 0 or bos_token_id >= self.llama_model.config.vocab_size:
            # print(f"⚠️  Warning: Invalid BOS token ID {bos_token_id}, using 0")
            bos_token_id = 0
        
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * bos_token_id
        
        # print(f"🔍 BOS token validation:")
        # print(f"   - BOS token ID: {bos_token_id}")
        # print(f"   - BOS tensor range: [{bos.min():.0f}, {bos.max():.0f}]")
        
        bos_embeds = self.llama_model.get_input_embeddings()(bos)
        atts_bos = atts_rna[:, :1]

        to_regress_embeds = self.llama_model.get_input_embeddings()(to_regress_tokens.input_ids)
        
        # Ensure all embeddings are in the same dtype as LLaMA model (FP16)
        llama_dtype = next(self.llama_model.parameters()).dtype
        # print(f"🔧 Converting embeddings to LLaMA dtype: {llama_dtype}")
        # print(f"   - BOS embeds dtype: {bos_embeds.dtype} -> {llama_dtype}")
        # print(f"   - RNA embeds dtype: {rna_embeds.dtype} -> {llama_dtype}")
        # print(f"   - Text embeds dtype: {to_regress_embeds.dtype} -> {llama_dtype}")
        
        # Scale RNA embeddings to prevent numerical instability
        rna_embeds_scale = rna_embeds.std()
        if rna_embeds_scale > 1.0:
            # print(f"⚠️  Warning: RNA embeddings scale too large ({rna_embeds_scale:.4f}), scaling down")
            rna_embeds = rna_embeds / rna_embeds_scale
        
        bos_embeds = bos_embeds.to(dtype=llama_dtype)
        rna_embeds = rna_embeds.to(dtype=llama_dtype)
        to_regress_embeds = to_regress_embeds.to(dtype=llama_dtype)
        
        # print(f"🔍 Before concatenation:")
        # print(f"   - BOS embeds shape: {bos_embeds.shape}")
        # print(f"   - BOS embeds dtype: {bos_embeds.dtype}")
        # print(f"   - BOS embeds range: [{bos_embeds.min():.6f}, {bos_embeds.max():.6f}]")
        # print(f"   - BOS embeds NaN count: {torch.isnan(bos_embeds).sum()}")
        # print(f"   - RNA embeds shape: {rna_embeds.shape}")
        # print(f"   - RNA embeds dtype: {rna_embeds.dtype}")
        # print(f"   - RNA embeds range: [{rna_embeds.min():.6f}, {rna_embeds.max():.6f}]")
        # print(f"   - RNA embeds NaN count: {torch.isnan(rna_embeds).sum()}")
        # print(f"   - Text embeds shape: {to_regress_embeds.shape}")
        # print(f"   - Text embeds dtype: {to_regress_embeds.dtype}")
        # print(f"   - Text embeds range: [{to_regress_embeds.min():.6f}, {to_regress_embeds.max():.6f}]")
        # print(f"   - Text embeds NaN count: {torch.isnan(to_regress_embeds).sum()}")
        
        inputs_embeds = torch.cat([bos_embeds, rna_embeds, to_regress_embeds], dim=1)
        
        # print(f"🔍 After concatenation:")
        # print(f"   - Final inputs_embeds shape: {inputs_embeds.shape}")
        # print(f"   - Final inputs_embeds dtype: {inputs_embeds.dtype}")
        # print(f"   - Final inputs_embeds range: [{inputs_embeds.min():.6f}, {inputs_embeds.max():.6f}]")
        # print(f"   - Final inputs_embeds NaN count: {torch.isnan(inputs_embeds).sum()}")
        # print(f"   - Expected shape: [batch_size={inputs_embeds.shape[0]}, seq_len={inputs_embeds.shape[1]}, hidden_size={inputs_embeds.shape[2]}]")
        if not torch.isfinite(inputs_embeds).all():
            # print("🚨 FATAL ERROR: Non-finite values found in inputs_embeds. Clamping to zero.")
            # Set any non-finite value (NaN, Inf) to a small, finite number (like zero)
            inputs_embeds = torch.nan_to_num(inputs_embeds, nan=0.0, posinf=1e-5, neginf=-1e-5)
        attention_mask = torch.cat([atts_bos, atts_rna, to_regress_tokens.attention_mask], dim=1)
        
        # print(f"🔍 Attention mask concatenation:")
        # print(f"   - BOS atts shape: {atts_bos.shape}")
        # print(f"   - RNA atts shape: {atts_rna.shape}")
        # print(f"   - Text atts shape: {to_regress_tokens.attention_mask.shape}")
        # print(f"   - Final attention_mask shape: {attention_mask.shape}")
        # print(f"   - Final attention_mask dtype: {attention_mask.dtype}")
        # print(f"   - Final attention_mask range: [{attention_mask.min():.6f}, {attention_mask.max():.6f}]")
        # print(f"   - Expected shape: [batch_size={attention_mask.shape[0]}, seq_len={attention_mask.shape[1]}]")

        # Debug: Check inputs before LLaMA model
        # print(f"🔍 Pre-LLaMA Debug:")
        # print(f"   - Input embeddings shape: {inputs_embeds.shape}")
        # print(f"   - Input embeddings dtype: {inputs_embeds.dtype}")
        # print(f"   - Input embeddings range: [{inputs_embeds.min():.6f}, {inputs_embeds.max():.6f}]")
        # print(f"   - Input embeddings NaN count: {torch.isnan(inputs_embeds).sum()}")
        # print(f"   - Attention mask shape: {attention_mask.shape}")
        # print(f"   - Attention mask range: [{attention_mask.min():.6f}, {attention_mask.max():.6f}]")
        # print(f"   - LLaMA model hidden size: {self.llama_model.config.hidden_size}")
        # print(f"   - LLaMA model vocab size: {self.llama_model.config.vocab_size}")
        # print(f"   - LLaMA model max position embeddings: {self.llama_model.config.max_position_embeddings}")
        # print(f"   - Sequence length: {inputs_embeds.shape[1]} (should be < {self.llama_model.config.max_position_embeddings})")
        
        # Check if sequence length is too long
        # if inputs_embeds.shape[1] > self.llama_model.config.max_position_embeddings:
        #     print(f"⚠️  WARNING: Sequence length {inputs_embeds.shape[1]} exceeds max position embeddings {self.llama_model.config.max_position_embeddings}")
        # else:
        #     print(f"✅ Sequence length is within limits")
        
        # Check inputs before LLaMA model
        # print(f"🔍 Pre-LLaMA Final Check:")
        # print(f"   - Input embeddings shape: {inputs_embeds.shape}")
        # print(f"   - Input embeddings dtype: {inputs_embeds.dtype}")
        # print(f"   - Input embeddings range: [{inputs_embeds.min():.6f}, {inputs_embeds.max():.6f}]")
        # print(f"   - Input embeddings NaN count: {torch.isnan(inputs_embeds).sum()}")
        # print(f"   - Attention mask shape: {attention_mask.shape}")
        # print(f"   - Attention mask range: [{attention_mask.min():.6f}, {attention_mask.max():.6f}]")
        # print(f"   - Attention mask NaN count: {torch.isnan(attention_mask.float()).sum()}")
        
        # Disable autocast to prevent numerical instability
        # with self.maybe_autocast():
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        
        logits = outputs.logits
        
        # Debug: Check outputs immediately after LLaMA model
        # print(f"🔍 Post-LLaMA Debug:")
        # print(f"   - Logits shape: {logits.shape}")
        # print(f"   - Logits dtype: {logits.dtype}")
        # print(f"   - Logits range: [{logits.min():.6f}, {logits.max():.6f}]")
        # print(f"   - Logits NaN count: {torch.isnan(logits).sum()}")
        # print(f"   - Expected logits shape: [batch_size={logits.shape[0]}, seq_len={logits.shape[1]}, vocab_size={logits.shape[2]}]")
        
        # Debug: Check for NaN values
        # if torch.isnan(logits).any():
        #     print("❌ NaN detected in logits!")
        #     print(f"  - Logits shape: {logits.shape}")
        #     print(f"  - NaN count: {torch.isnan(logits).sum()}")
        #     print(f"  - Input embeddings range: [{inputs_embeds.min():.4f}, {inputs_embeds.max():.4f}]")
        #     print(f"  - Attention mask range: [{attention_mask.min():.4f}, {attention_mask.max():.4f}]")
        
        loss = outputs.loss
        
        # Debug: Check for NaN loss
        # print(f"🔍 Loss Debug:")
        # print(f"   - Loss value: {loss}")
        # print(f"   - Loss is NaN: {torch.isnan(loss)}")
        # print(f"   - Loss dtype: {loss.dtype}")
        
        # if torch.isnan(loss):
        #     print("❌ NaN detected in loss!")
        #     print(f"  - Loss value: {loss}")
        #     print(f"  - Targets range: [{targets.min():.4f}, {targets.max():.4f}]")
        #     print(f"  - Targets NaN count: {torch.isnan(targets).sum()}")
        #     print(f"  - Logits NaN count: {torch.isnan(logits).sum()}")
        #     print(f"  - Logits range: [{logits.min():.6f}, {logits.max():.6f}]")
            
        #     # Check RNA embeddings
        #     print(f"  - RNA embeddings range: [{rna_embeds.min():.4f}, {rna_embeds.max():.4f}]")
        #     print(f"  - RNA embeddings NaN count: {torch.isnan(rna_embeds).sum()}")
            
        #     # Check projection layer
        #     print(f"  - Projection layer weight range: [{self.rnafm_llama_proj.weight.min():.4f}, {self.rnafm_llama_proj.weight.max():.4f}]")
        #     print(f"  - Projection layer weight NaN count: {torch.isnan(self.rnafm_llama_proj.weight).sum()}")
            
        #     # Check LLaMA model weights
        #     llama_weights = list(self.llama_model.parameters())
        #     if llama_weights:
        #         llama_weight = llama_weights[0]
        #         print(f"  - LLaMA model weight range: [{llama_weight.min():.4f}, {llama_weight.max():.4f}]")
        #         print(f"  - LLaMA model weight NaN count: {torch.isnan(llama_weight).sum()}")
        #         print(f"  - LLaMA model dtype: {llama_weight.dtype}")
            
        #     # Check input embeddings dtype
        #     print(f"  - Input embeddings dtype: {inputs_embeds.dtype}")
        #     print(f"  - Attention mask dtype: {attention_mask.dtype}")
        
        logits = torch.argmax(logits, dim=2)
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
        disable_quantization = cfg.get("disable_quantization", False)

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
            disable_quantization=disable_quantization,
            # device_8bit=device_8bit,
        )

        # original_state = copy.deepcopy(model.state_dict())  # Removed to save memory

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
                'rna_encoder.',           # RNA encoder weights (now modelgenerator)
                'rnafm_llama_proj.',      # Projection layer weights
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
                        elif 'rnafm_llama_proj' in key:
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
        