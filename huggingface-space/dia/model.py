import time
from enum import Enum
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from .audio import apply_audio_delay, build_delay_indices, build_revert_indices, revert_audio_delay
from .config import DiaConfig
from .layers import DiaModel
from .state import DecoderInferenceState, DecoderOutput, EncoderInferenceState


DEFAULT_SAMPLE_RATE = 44100
SAMPLE_RATE_RATIO = 512


def _get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sample_next_token(
    logits_BCxV: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int | None,
    audio_eos_value: int,
) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits_BCxV, dim=-1)

    logits_BCxV = logits_BCxV / temperature

    if audio_eos_value is not None and audio_eos_value >= 0:
        top_logit_indices_BC = torch.argmax(logits_BCxV, dim=-1)
        eos_not_highest_mask_BC = top_logit_indices_BC != audio_eos_value
        mask_eos_unless_highest_BCxV = torch.zeros_like(logits_BCxV, dtype=torch.bool)
        mask_eos_unless_highest_BCxV[eos_not_highest_mask_BC, audio_eos_value] = True
        logits_BCxV = logits_BCxV.masked_fill(mask_eos_unless_highest_BCxV, -torch.inf)
        eos_highest_mask_BC = top_logit_indices_BC == audio_eos_value
        mask_eos_highest_BCxV = torch.zeros_like(logits_BCxV, dtype=torch.bool)
        mask_eos_highest_BCxV[eos_highest_mask_BC, :audio_eos_value] = True
        logits_BCxV = logits_BCxV.masked_fill(mask_eos_highest_BCxV, -torch.inf)

    if top_k is not None:
        _, top_k_indices_BCxV = torch.topk(logits_BCxV, k=top_k, dim=-1)
        mask = torch.ones_like(logits_BCxV, dtype=torch.bool)
        mask = mask.scatter(dim=-1, index=top_k_indices_BCxV, value=False)
        logits_BCxV = logits_BCxV.masked_fill(mask, -torch.inf)

    if top_p < 1.0:
        probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
        sorted_probs_BCxV, sorted_indices_BCxV = torch.sort(probs_BCxV, dim=-1, descending=True)
        cumulative_probs_BCxV = torch.cumsum(sorted_probs_BCxV, dim=-1)

        sorted_indices_to_remove_BCxV = cumulative_probs_BCxV > top_p
        sorted_indices_to_remove_BCxV = torch.roll(sorted_indices_to_remove_BCxV, shifts=1, dims=-1)
        sorted_indices_to_remove_BCxV[..., 0] = torch.zeros_like(sorted_indices_to_remove_BCxV[..., 0])

        indices_to_remove_BCxV = torch.zeros_like(sorted_indices_to_remove_BCxV)
        indices_to_remove_BCxV = indices_to_remove_BCxV.scatter(
            dim=-1, index=sorted_indices_BCxV, src=sorted_indices_to_remove_BCxV
        )
        logits_BCxV = logits_BCxV.masked_fill(indices_to_remove_BCxV, -torch.inf)

    final_probs_BCxV = torch.softmax(logits_BCxV, dim=-1)

    sampled_indices_BC = torch.multinomial(final_probs_BCxV, num_samples=1)
    sampled_indices_C = sampled_indices_BC.squeeze(-1)
    return sampled_indices_C


class ComputeDtype(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    def to_dtype(self) -> torch.dtype:
        if self == ComputeDtype.FLOAT32:
            return torch.float32
        elif self == ComputeDtype.FLOAT16:
            return torch.float16
        elif self == ComputeDtype.BFLOAT16:
            return torch.bfloat16
        else:
            raise ValueError(f"Unsupported compute dtype: {self}")


class Dia:
    def __init__(
        self,
        config: DiaConfig,
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
        load_dac: bool = True,
    ):
        """Initializes the Dia model.

        Args:
            config: The configuration object for the model.
            compute_dtype: The computation dtype to use.
            device: The device to load the model onto. If None, will automatically select the best available device.
            load_dac: Whether to load the DAC model.

        Raises:
            RuntimeError: If there is an error loading the DAC model.
        """
        super().__init__()
        self.config = config
        self.device = device if device is not None else _get_default_device()
        if isinstance(compute_dtype, str):
            compute_dtype = ComputeDtype(compute_dtype)
        self.compute_dtype = compute_dtype.to_dtype()
        self.model: DiaModel = DiaModel(config, self.compute_dtype)
        self.dac_model = None
        self._compiled_step = None
        self.load_dac = load_dac

        if not self.load_dac:
            print("Warning: DAC model will not be loaded. This is not recommended.")

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

    @classmethod
    def from_local(
        cls,
        config_path: str,
        checkpoint_path: str,
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
        load_dac: bool = True,
    ) -> "Dia":
        """Loads the Dia model from local configuration and checkpoint files.

        Args:
            config_path: Path to the configuration JSON file.
            checkpoint_path: Path to the model checkpoint (.pth) file.
            compute_dtype: The computation dtype to use.
            device: The device to load the model onto. If None, will automatically select the best available device.
            load_dac: Whether to load the DAC model.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If the config or checkpoint file is not found.
            RuntimeError: If there is an error loading the checkpoint.
        """
        config = DiaConfig.load(config_path)
        if config is None:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        dia = cls(config, compute_dtype, device, load_dac)

        try:
            state_dict = torch.load(checkpoint_path, map_location=dia.device)
            dia.model.load_state_dict(state_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint from {checkpoint_path}") from e

        dia.model.to(dia.device)
        dia.model.eval()
        if load_dac:
            dia._load_dac_model()
        return dia

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "nari-labs/Dia-1.6B-0626",
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
        load_dac: bool = True,
    ) -> "Dia":
        """Loads the Dia model from a Hugging Face Hub repository.

        Downloads the configuration and checkpoint files from the specified
        repository ID and then loads the model.

        Args:
            model_name: The Hugging Face Hub repository ID (e.g., "nari-labs/Dia-1.6B-0626").
            compute_dtype: The computation dtype to use.
            device: The device to load the model onto. If None, will automatically select the best available device.
            load_dac: Whether to load the DAC model.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If config or checkpoint download/loading fails.
            RuntimeError: If there is an error loading the checkpoint.
        """
        if isinstance(compute_dtype, str):
            compute_dtype = ComputeDtype(compute_dtype)

        # Load model directly using DiaModel's from_pretrained which handles HF download
        try:
            loaded_model = DiaModel.from_pretrained(model_name, compute_dtype=compute_dtype.to_dtype())
        except Exception as e:
            raise RuntimeError(f"Error loading model from Hugging Face Hub ({model_name})") from e

        config = loaded_model.config  # Get config from the loaded model
        dia = cls(config, compute_dtype, device, load_dac)

        dia.model = loaded_model  # Assign the already loaded model
        dia.model.to(dia.device)
        dia.model.eval()
        if load_dac:
            dia._load_dac_model()
        return dia

    def _load_dac_model(self):
        """Loads the Descript Audio Codec (DAC) model.

        Downloads the DAC model if necessary and loads it onto the specified device.
        Sets the DAC model to evaluation mode.

        Raises:
            RuntimeError: If downloading or loading the DAC model fails.
        """
        import dac

        try:
            dac_model_path = dac.utils.download()
            dac_model = dac.DAC.load(dac_model_path).to(self.device)
            dac_model.eval()  # Ensure DAC is in eval mode
        except Exception as e:
            raise RuntimeError("Failed to load DAC model") from e
        self.dac_model = dac_model

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encodes the input text string into a tensor of token IDs using byte-level encoding.

        Special tokens [S1] and [S2] are replaced by their byte values. The resulting
        sequence is truncated to the maximum configured text length.

        Args:
            text: The input text string.

        Returns:
            A tensor containing the encoded byte token IDs.
        """
        max_len = self.config.encoder_config.max_position_embeddings

        byte_text = text.encode("utf-8")
        # Replace special tokens with their byte values if needed by the specific tokenizer/config
        # Assuming byte values 1 and 2 are correct placeholders based on original code
        replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
        text_tokens = list(replaced_bytes)
        return torch.tensor(
            text_tokens[:max_len],
            dtype=torch.long,
            device=self.device,
        )

    def _pad_text_input(self, text_tokens: list[torch.Tensor]) -> torch.Tensor:
        """Pads the text input to the maximum length."""
        text_pad_value = 0
        max_len = self.config.encoder_config.max_position_embeddings
        batch_size = len(text_tokens)

        src_tokens = torch.full(
            (batch_size, 1, max_len),
            fill_value=text_pad_value,
            dtype=torch.long,
            device=self.device,
        )
        for i in range(batch_size):
            current_len = len(text_tokens[i])
            src_tokens[i, 0, :current_len] = text_tokens[i]
        return src_tokens

    def _prepare_audio_prompt(self, audio_prompts: list[torch.Tensor | None]) -> tuple[torch.Tensor, list[int]]:
        """Prepares the audio prompt tensor for the decoder.

        Handles padding, adds the beginning-of-sequence (BOS) token, applies the
        delay pattern, and determines the number of prefill steps for each item
        in the batch.

        Args:
            audio_prompts: A list of audio prompt tensors (encoded DAC frames) or None.
                           Each tensor should have shape [T, C].

        Returns:
            A tuple containing:
                - delayed_batch (torch.Tensor): The prepared audio prompt tensor with
                  delays applied, shape [B, T_max_padded, C].
                - prefill_steps (list[int]): A list containing the number of valid
                  tokens (including BOS) for each prompt in the batch.
        """
        num_channels = self.config.decoder_config.num_channels
        audio_bos_value = self.config.bos_token_id
        delay_pattern = self.config.delay_pattern
        max_delay_pattern = max(delay_pattern)
        batch_size = len(audio_prompts)

        max_len = max(p.shape[0] if p is not None else 0 for p in audio_prompts) + max_delay_pattern
        prefill_steps = []

        prefill = torch.full(
            (batch_size, max_len, num_channels),
            fill_value=-1,
            dtype=torch.int,
            device=self.device,
        )

        prefill[:, 0, :] = audio_bos_value

        for i in range(batch_size):
            prompt = audio_prompts[i]
            if prompt is not None:
                prompt = prompt.to(device=self.device, dtype=torch.int)
                prefill[i, 1 : prompt.shape[0] + 1, :] = prompt
                prefill_steps.append(prompt.shape[0] + 1)
            else:
                prefill_steps.append(1)

        delay_precomp = build_delay_indices(
            B=batch_size,
            T=max_len,
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        delayed_batch = apply_audio_delay(
            audio_BxTxC=prefill,
            pad_value=-1,
            bos_value=audio_bos_value,
            precomp=delay_precomp,
        )

        return delayed_batch, prefill_steps

    def _prepare_generation(
        self,
        text: torch.Tensor,
        audio_prompts: list[torch.Tensor | None],
        max_tokens: int | None = None,
        attn_fn: Callable = F.scaled_dot_product_attention,
    ):
        """Initializes the model state for generation.

        Encodes the text input (conditional and unconditional), prepares the
        encoder and decoder states (including KV caches and cross-attention),
        prepares the audio prompt, and performs the initial decoder prefill steps
        based on the audio prompts.

        Args:
            text: The padded text input tensor, shape [B, 1, T_text].
            audio_prompts: A list of prepared audio prompt tensors or None.

        Returns:
            A tuple containing:
                - dec_state (DecoderInferenceState): The initialized decoder state.
                - dec_output (DecoderOutput): The initialized decoder output manager,
                  containing the prefilled audio tokens.
        """
        batch_size = text.shape[0]

        enc_input_uncond = torch.zeros_like(text)
        enc_input_cond = text
        stacked_inputs = torch.stack([enc_input_uncond, enc_input_cond], dim=1)
        enc_input = stacked_inputs.view(2 * batch_size, -1)

        enc_state = EncoderInferenceState.new(self.config, enc_input_cond)
        encoder_out = self.model.encoder(enc_input, enc_state)

        dec_cross_attn_cache = self.model.decoder.precompute_cross_attn_cache(encoder_out)
        dec_state = DecoderInferenceState.new(
            self.config,
            enc_state,
            encoder_out,
            dec_cross_attn_cache,
            self.compute_dtype,
            max_generation_length=max_tokens,
        )
        prefill, prefill_steps = self._prepare_audio_prompt(audio_prompts)

        dec_output = DecoderOutput.new(batch_size, self.config, self.device)
        dec_output.prefill(prefill, prefill_steps)

        dec_step = min(prefill_steps) - 1
        if dec_step > 0:
            dec_state.prepare_step(0, dec_step)
            tokens_BxTxC = dec_output.get_tokens_at(0, dec_step).repeat_interleave(2, dim=0)
            self.model.decoder.forward(tokens_BxTxC, dec_state)

        return dec_state, dec_output

    def _decoder_step(
        self,
        tokens_Bx1xC: torch.Tensor,
        dec_state: DecoderInferenceState,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        top_k: int,
        current_idx: int,
    ) -> torch.Tensor:
        """Performs a single step of the decoder inference.

        Takes the tokens from the previous step, runs them through the decoder
        (for both conditional and unconditional paths), applies classifier-free
        guidance (CFG), samples the next token using temperature, top-p, and top-k
        sampling, and applies constraints (e.g., preventing EOS in certain channels).

        Args:
            tokens_Bx1xC: The input tokens for the current step, shape [2*B, 1, C].
                         Repeated for CFG (unconditional and conditional).
            dec_state: The current state of the decoder (KV caches, etc.).
            cfg_scale: The scale factor for classifier-free guidance.
            temperature: The temperature for sampling.
            top_p: The cumulative probability threshold for top-p sampling.
            top_k: The number of top logits to consider for top-k sampling.
            current_idx: The current generation step index.

        Returns:
            torch.Tensor: The sampled next tokens for each item in the batch,
                          shape [B, C].
        """
        B = tokens_Bx1xC.shape[0] // 2

        audio_eos_value = self.config.eos_token_id
        logits_Bx1xCxV = self.model.decoder.decode_step(tokens_Bx1xC, dec_state, current_idx)

        logits_last_2BxCxV = logits_Bx1xCxV[:, -1]
        logits_last_Bx2xCxV = logits_last_2BxCxV.view(B, 2, *logits_last_2BxCxV.shape[1:])

        uncond_logits_BxCxV = logits_last_Bx2xCxV[:, 0, :, :]  # Shape [B, C, V]
        cond_logits_BxCxV = logits_last_Bx2xCxV[:, 1, :, :]  # Shape [B, C, V]
        logits_BxCxV = cond_logits_BxCxV + cfg_scale * (cond_logits_BxCxV - uncond_logits_BxCxV)

        _, top_k_indices_BxCxk = torch.topk(logits_BxCxV, k=top_k, dim=-1)
        mask_BxCxV = torch.ones_like(logits_BxCxV, dtype=torch.bool)
        mask_BxCxV = mask_BxCxV.scatter(dim=-1, index=top_k_indices_BxCxk, value=False)
        logits_BxCxV = cond_logits_BxCxV.masked_fill(mask_BxCxV, -torch.inf)

        logits_BxCxV[:, :, audio_eos_value + 1 :] = torch.full_like(
            logits_BxCxV[:, :, audio_eos_value + 1 :],
            fill_value=-torch.inf,
        )

        num_channels = self.config.decoder_config.num_channels

        # ... keep existing code (rest of the method implementation)
        next_tokens_BxC = torch.zeros(B, num_channels, dtype=torch.long, device=self.device)
        
        for ch in range(num_channels):
            logits_BxV = logits_BxCxV[:, ch, :]
            next_tokens_BxC[:, ch] = _sample_next_token(
                logits_BxV, temperature, top_p, top_k, audio_eos_value
            )

        return next_tokens_BxC

    def _generate_output(self, dec_output: DecoderOutput) -> list[np.ndarray]:
        """Converts the generated audio codes back to audio waveforms using DAC.

        Args:
            dec_output: The decoder output containing the generated audio tokens.

        Returns:
            A list of audio waveforms (as numpy arrays) for each item in the batch.
        """
        if self.dac_model is None:
            raise RuntimeError("DAC model is not loaded. Cannot generate audio output.")

        generated_codes_BxTxC = dec_output.get_generated_tokens()
        batch_size = generated_codes_BxTxC.shape[0]
        audio_list = []

        for i in range(batch_size):
            codes_TxC = generated_codes_BxTxC[i]
            audio_waveform = self._decode(codes_TxC)
            audio_list.append(audio_waveform)

        return audio_list

    def _encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encodes audio using the DAC model.

        Args:
            audio: Audio tensor with shape [1, 1, T] where T is the number of samples.

        Returns:
            Encoded audio codes with shape [T_frames, C] where T_frames is the number
            of frames and C is the number of codebook channels.
        """
        if self.dac_model is None:
            raise RuntimeError("DAC model is not loaded. Cannot encode audio.")

        with torch.no_grad():
            codes, _, _ = self.dac_model.encode(audio)
            codes = codes.squeeze(0)  # Remove batch dimension: [C, T_frames]
            codes = codes.transpose(0, 1)  # Transpose to [T_frames, C]

        return codes

    def _decode(self, audio_codes: torch.Tensor) -> np.ndarray:
        """Decodes audio codes using the DAC model.

        Args:
            audio_codes: Audio codes tensor with shape [T_frames, C].

        Returns:
            Decoded audio as a numpy array.
        """
        if self.dac_model is None:
            raise RuntimeError("DAC model is not loaded. Cannot decode audio.")

        # Apply delay reversal
        num_channels = self.config.decoder_config.num_channels
        delay_pattern = self.config.delay_pattern
        
        delay_revert_precomp = build_revert_indices(
            B=1,
            T=audio_codes.shape[0],
            C=num_channels,
            delay_pattern=delay_pattern,
        )
        
        codes_with_delay_1xTxC = audio_codes.unsqueeze(0)  # Add batch dimension
        reverted_codes_1xTxC = revert_audio_delay(codes_with_delay_1xTxC, delay_revert_precomp)
        reverted_codes_TxC = reverted_codes_1xTxC.squeeze(0)  # Remove batch dimension
        
        # Transpose for DAC: [T_frames, C] -> [C, T_frames]
        codes_CxT = reverted_codes_TxC.transpose(0, 1)
        codes_1xCxT = codes_CxT.unsqueeze(0)  # Add batch dimension: [1, C, T_frames]

        with torch.no_grad():
            audio_1x1xT = self.dac_model.decode(codes_1xCxT)
            audio_T = audio_1x1xT.squeeze()  # Remove batch and channel dimensions

        return audio_T.cpu().numpy()

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Loads and encodes an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Encoded audio codes with shape [T_frames, C].
        """
        audio, sr = torchaudio.load(audio_path)
        
        # Resample to DAC's expected sample rate if necessary
        if sr != DEFAULT_SAMPLE_RATE:
            audio = torchaudio.transforms.Resample(sr, DEFAULT_SAMPLE_RATE)(audio)
        
        # Ensure single channel and add batch dimension
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        audio = audio.unsqueeze(0)  # Shape: [1, 1, T]
        
        audio = audio.to(self.device)
        return self._encode(audio)

    def save_audio(self, path: str, audio: np.ndarray):
        """Saves an audio waveform to a file.

        Args:
            path: Output file path.
            audio: Audio waveform as a numpy array.
        """
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # Add channel dimension
        torchaudio.save(path, audio_tensor, DEFAULT_SAMPLE_RATE)

    def generate(
        self,
        text: str | list[str],
        max_tokens: int = 3072,
        cfg_scale: float = 3.0,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        audio_prompts: list[torch.Tensor | None] | None = None,
        compile_step: bool = False,
    ) -> list[np.ndarray]:
        """Generates audio from text using the Dia model.

        Args:
            text: Input text string or list of text strings.
            max_tokens: Maximum number of tokens to generate.
            cfg_scale: Scale factor for classifier-free guidance.
            temperature: Temperature for sampling.
            top_p: Cumulative probability threshold for nucleus sampling.
            top_k: Number of top tokens to consider for sampling.
            audio_prompts: Optional list of audio prompt tensors.
            compile_step: Whether to use torch.compile for faster inference.

        Returns:
            A list of generated audio waveforms as numpy arrays.
        """
        if isinstance(text, str):
            text = [text]
        
        batch_size = len(text)
        
        if audio_prompts is None:
            audio_prompts = [None] * batch_size
        elif len(audio_prompts) != batch_size:
            raise ValueError("Number of audio prompts must match batch size")

        # Encode text inputs
        text_tokens = [self._encode_text(t) for t in text]
        text_tensor = self._pad_text_input(text_tokens)

        # Initialize generation state
        dec_state, dec_output = self._prepare_generation(
            text_tensor, audio_prompts, max_tokens
        )

        # Compile decoder step if requested
        if compile_step and self._compiled_step is None:
            self._compiled_step = torch.compile(self._decoder_step)
        
        decoder_step_fn = self._compiled_step if compile_step else self._decoder_step

        # Generation loop
        max_generated_tokens = max_tokens
        for step in range(max_generated_tokens):
            current_tokens = dec_output.get_current_tokens().repeat_interleave(2, dim=0)
            
            next_tokens = decoder_step_fn(
                current_tokens,
                dec_state,
                cfg_scale,
                temperature,
                top_p,
                top_k,
                step
            )
            
            if not dec_output.append_tokens(next_tokens):
                break  # All sequences have reached EOS

        # Convert to audio
        audio_outputs = self._generate_output(dec_output)
        return audio_outputs