from typing import List, Optional, Union, Dict
import tqdm
import torch
import torch.nn.functional as F
import openvino as ov
from pathlib import Path
from bark.generation import load_model, codec_decode, _flatten_codebooks
import numpy as np
import os
from bark.generation import (
    _load_history_prompt,
    _tokenize,
    _normalize_whitespace,
    TEXT_PAD_TOKEN,
    TEXT_ENCODING_OFFSET,
    SEMANTIC_VOCAB_SIZE,
    SEMANTIC_PAD_TOKEN,
    SEMANTIC_INFER_TOKEN,
    COARSE_RATE_HZ,
    SEMANTIC_RATE_HZ,
    N_COARSE_CODEBOOKS,
    COARSE_INFER_TOKEN,
    CODEBOOK_SIZE,
    N_FINE_CODEBOOKS,
    COARSE_SEMANTIC_PAD_TOKEN,
)
def SpeechSynthesis(device):
    module_path = Path(os.path.split(__file__)[0])
    models_dir = module_path/Path("models")
    models_dir.mkdir(exist_ok=True)
    text_use_small = True
    coarse_use_small=True
    text_encoder = load_model(
        model_type="text", use_gpu=False, use_small=text_use_small, force_reload=False
    )
    text_encoder_model = text_encoder["model"]
    tokenizer = text_encoder["tokenizer"]
    text_model_suffix = "_small" if text_use_small else ""
    text_model_dir = models_dir / f"text_encoder{text_model_suffix}"
    text_model_dir.mkdir(exist_ok=True)
    text_encoder_path1 = text_model_dir / "bark_text_encoder_1.xml"
    text_encoder_path0 = text_model_dir / "bark_text_encoder_0.xml"


    coarse_model = load_model(
        model_type="coarse", use_gpu=False, use_small=coarse_use_small, force_reload=False, 
    )
    coarse_model_suffix = "_small" if coarse_use_small else ""
    coarse_model_dir = models_dir / f"coarse{coarse_model_suffix}"
    coarse_model_dir.mkdir(exist_ok=True)
    coarse_encoder_path = coarse_model_dir / "bark_coarse_encoder.xml"

    fine_use_small = False

    fine_model = load_model(model_type="fine", use_gpu=False, use_small=fine_use_small, force_reload=False)

    fine_model_suffix = "_small" if fine_use_small else ""
    fine_model_dir = models_dir / f"fine_model{fine_model_suffix}"
    fine_model_dir.mkdir(exist_ok=True)
    fine_feature_extractor_path = fine_model_dir / "bark_fine_feature_extractor.xml"

    core = ov.Core()
    class TextEncoderModel(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, idx, past_kv=None):
            return self.encoder(idx, merge_context=True, past_kv=past_kv, use_cache=True)
    class OVBarkTextEncoder:
        def __init__(self, core, device, model_path1, model_path2):

            if not model_path1.exists() or not model_path2.exists():
                text_encoder_exportable = TextEncoderModel(text_encoder_model)
                ov_model = ov.convert_model(
                    text_encoder_exportable, example_input=torch.ones((1, 513), dtype=torch.int64)
                )
                ov.save_model(ov_model, model_path1)

                self.compiled_model1=ov_model

                logits, kv_cache = text_encoder_exportable(torch.ones((1, 513), dtype=torch.int64))
                ov_model = ov.convert_model(
                    text_encoder_exportable,
                    example_input=(torch.ones((1, 1), dtype=torch.int64), kv_cache),
                )
                ov.save_model(ov_model, model_path2)
                self.compiled_model2=ov_model
            else:
                self.compiled_model1 = core.compile_model(model_path1, device)
                self.compiled_model2 = core.compile_model(model_path2, device)

        def __call__(self, input_ids, past_kv=None):
            if past_kv is None:
                outputs = self.compiled_model1(input_ids, share_outputs=True)
            else:
                outputs = self.compiled_model2([input_ids, *past_kv], share_outputs=True)
            logits, kv_cache = self.postprocess_outputs(outputs, past_kv is None)
            return logits, kv_cache

        def postprocess_outputs(self, outs, is_first_stage):
            net_outs = (
                self.compiled_model1.outputs
                if is_first_stage
                else self.compiled_model2.outputs
            )
            logits = outs[net_outs[0]]
            kv_cache = []
            for out_tensor in net_outs[1:]:
                kv_cache.append(outs[out_tensor])
            return logits, kv_cache
    class OVBarkEncoder:
        def __init__(self, core, device, model_path):


            if not coarse_encoder_path.exists():
                coarse_encoder_exportable = CoarseEncoderModel(coarse_model)
                logits, kv_cache = coarse_encoder_exportable(
                    torch.ones((1, 886), dtype=torch.int64)
                )
                ov_model = ov.convert_model(
                    coarse_encoder_exportable,
                    example_input=(torch.ones((1, 1), dtype=torch.int64), kv_cache),
                )
                ov.save_model(ov_model, coarse_encoder_path)
                
                self.compiled_model=ov_model
            else:
                self.compiled_model = core.compile_model(model_path, device)

        def __call__(self, idx, past_kv=None):
            if past_kv is None:
                past_kv = self._init_past_kv()
            outs = self.compiled_model([idx, *past_kv], share_outputs=True)
            return self.postprocess_outputs(outs)

        def postprocess_outputs(self, outs):
            net_outs = self.compiled_model.outputs
            logits = outs[net_outs[0]]
            kv_cache = []
            for out_tensor in net_outs[1:]:
                kv_cache.append(outs[out_tensor])
            return logits, kv_cache

        def _init_past_kv(self):
            inputs = []
            for input_t in self.compiled_model.inputs[1:]:
                input_shape = input_t.partial_shape
                input_shape[0] = 1
                input_shape[2] = 0
                inputs.append(ov.Tensor(ov.Type.f32, input_shape.get_shape()))
            return inputs
    class CoarseEncoderModel(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, idx, past_kv=None):
            return self.encoder(idx, past_kv=past_kv, use_cache=True)
    class OVBarkFineEncoder:
        def __init__(self, core, device, fine_feature_extractor_path, num_lm_heads=7):


            if not fine_feature_extractor_path.exists():
                lm_heads = fine_model.lm_heads
                fine_feature_extractor = FineModel(fine_model)
                feature_extractor_out = fine_feature_extractor(
                    3, torch.zeros((1, 1024, 8), dtype=torch.int32)
                )
                ov_model = ov.convert_model(
                    fine_feature_extractor,
                    example_input=(
                        torch.ones(1, dtype=torch.long),
                        torch.zeros((1, 1024, 8), dtype=torch.long),
                    ),
                )
                ov.save_model(ov_model, fine_feature_extractor_path)
                for i, lm_head in enumerate(lm_heads):
                    ov.save_model(
                        ov.convert_model(lm_head, example_input=feature_extractor_out),
                        fine_model_dir / f"bark_fine_lm_{i}.xml",
                    )
                self.lm_heads = lm_heads
                self.feats_compiled_model=ov_model

            
            else:
                self.feats_compiled_model = core.compile_model(
                    fine_feature_extractor_path, device
                )
                lm_heads = []
                for i in range(num_lm_heads):
                    lm_heads.append(
                        core.compile_model(fine_model_dir / f"bark_fine_lm_{i}.xml", device)
                    )
                self.lm_heads = lm_heads

            self.feats_out = self.feats_compiled_model.output(0)

        def __call__(self, pred_idx, idx):
            feats = self.feats_compiled_model([ov.Tensor(pred_idx), ov.Tensor(idx)])[
                self.feats_out
            ]
            lm_id = pred_idx - 1
            logits = self.lm_heads[int(lm_id)](feats)[0]
            return logits
    class FineModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pred_idx, idx):
            b, t, codes = idx.size()
            pos = torch.arange(0, t, dtype=torch.long).unsqueeze(0)  # shape (1, t)

            # forward the GPT model itself
            tok_embs = [
                wte(idx[:, :, i]).unsqueeze(-1)
                for i, wte in enumerate(self.model.transformer.wtes)
            ]  # token embeddings of shape (b, t, n_embd)
            tok_emb = torch.cat(tok_embs, dim=-1)
            pos_emb = self.model.transformer.wpe(
                pos
            )  # position embeddings of shape (1, t, n_embd)
            x = tok_emb[:, :, :, : pred_idx + 1].sum(dim=-1)
            x = self.model.transformer.drop(x + pos_emb)
            for block in self.model.transformer.h:
                x = block(x)
            x = self.model.transformer.ln_f(x)
            return x
    ov_text_model = OVBarkTextEncoder(
        core, device, text_encoder_path0, text_encoder_path1
    )
    ov_coarse_model = OVBarkEncoder(core, device, coarse_encoder_path)
    ov_fine_model = OVBarkFineEncoder(core, device, fine_feature_extractor_path)





    def generate_text_semantic(
        text: str,
        history_prompt: List[str] = None,
        temp: float = 0.7,
        top_k: int = None,
        top_p: float = None,
        silent: bool = False,
        min_eos_p: float = 0.2,
        max_gen_duration_s: int = None,
        allow_early_stop: bool = True,
    ):
        """
        Generate semantic tokens from text.
        Args:
            text: text to be turned into audio
            history_prompt: history choice for audio cloning
            temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            top_k: top k number of probabilities for considering during generation
            top_p: top probabilities higher than p for considering during generation
            silent: disable progress bar
            min_eos_p: minimum probability to select end of string token
            max_gen_duration_s: maximum duration for generation in seconds
            allow_early_stop: allow to stop generation if maximum duration is not reached
        Returns:
            numpy semantic array to be fed into `semantic_to_waveform`

        """
        text = _normalize_whitespace(text)
        if history_prompt is not None:
            history_prompt = _load_history_prompt(history_prompt)
            semantic_history = history_prompt["semantic_prompt"]
        else:
            semantic_history = None
        encoded_text = (
            np.ascontiguousarray(_tokenize(tokenizer, text)) + TEXT_ENCODING_OFFSET
        )
        if len(encoded_text) > 256:
            p = round((len(encoded_text) - 256) / len(encoded_text) * 100, 1)
            print(f"warning, text too long, lopping of last {p}%")
            encoded_text = encoded_text[:256]
        encoded_text = np.pad(
            encoded_text,
            (0, 256 - len(encoded_text)),
            constant_values=TEXT_PAD_TOKEN,
            mode="constant",
        )
        if semantic_history is not None:
            semantic_history = semantic_history.astype(np.int64)
            # lop off if history is too long, pad if needed
            semantic_history = semantic_history[-256:]
            semantic_history = np.pad(
                semantic_history,
                (0, 256 - len(semantic_history)),
                constant_values=SEMANTIC_PAD_TOKEN,
                mode="constant",
            )
        else:
            semantic_history = np.array([SEMANTIC_PAD_TOKEN] * 256)
        x = np.hstack(
            [encoded_text, semantic_history, np.array([SEMANTIC_INFER_TOKEN])]
        ).astype(np.int64)[None]
        assert x.shape[1] == 256 + 256 + 1
        n_tot_steps = 768
        # custom tqdm updates since we don't know when eos will occur
        pbar = tqdm.tqdm(disable=silent, total=100)
        pbar_state = 0
        tot_generated_duration_s = 0
        kv_cache = None
        for n in range(n_tot_steps):
            if kv_cache is not None:
                x_input = x[:, [-1]]
            else:
                x_input = x
            logits, kv_cache = ov_text_model(ov.Tensor(x_input), kv_cache)
            relevant_logits = logits[0, 0, :SEMANTIC_VOCAB_SIZE]
            if allow_early_stop:
                relevant_logits = np.hstack(
                    (relevant_logits, logits[0, 0, [SEMANTIC_PAD_TOKEN]])
                )  # eos
            if top_p is not None:
                sorted_indices = np.argsort(relevant_logits)[::-1]
                sorted_logits = relevant_logits[sorted_indices]
                cumulative_probs = np.cumsum(F.softmax(sorted_logits))
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False
                relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                relevant_logits = torch.from_numpy(relevant_logits)
            if top_k is not None:
                relevant_logits = torch.from_numpy(relevant_logits)
                v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                relevant_logits[relevant_logits < v[-1]] = -float("Inf")
            probs = F.softmax(torch.from_numpy(relevant_logits) / temp, dim=-1)
            item_next = torch.multinomial(probs, num_samples=1)
            if allow_early_stop and (
                item_next == SEMANTIC_VOCAB_SIZE
                or (min_eos_p is not None and probs[-1] >= min_eos_p)
            ):
                # eos found, so break
                pbar.update(100 - pbar_state)
                break
            x = torch.cat((torch.from_numpy(x), item_next[None]), dim=1).numpy()
            tot_generated_duration_s += 1 / SEMANTIC_RATE_HZ
            if (
                max_gen_duration_s is not None
                and tot_generated_duration_s > max_gen_duration_s
            ):
                pbar.update(100 - pbar_state)
                break
            if n == n_tot_steps - 1:
                pbar.update(100 - pbar_state)
                break
            del logits, relevant_logits, probs, item_next
            req_pbar_state = np.min([100, int(round(100 * n / n_tot_steps))])
            if req_pbar_state > pbar_state:
                pbar.update(req_pbar_state - pbar_state)
            pbar_state = req_pbar_state
        pbar.close()
        out = x.squeeze()[256 + 256 + 1 :]
        return out


    def generate_audio(
        text: str,
        history_prompt: Optional[Union[Dict, str]] = None,
        text_temp: float = 0.7,
        waveform_temp: float = 0.7,
        silent: bool = False,
    ):
        """Generate audio array from input text.

        Args:
            text: text to be turned into audio
            history_prompt: history choice for audio cloning
            text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            silent: disable progress bar

        Returns:
            numpy audio array at sample frequency 24khz
        """
        semantic_tokens = text_to_semantic(
            text,
            history_prompt=history_prompt,
            temp=text_temp,
            silent=silent,
        )
        out = semantic_to_waveform(
            semantic_tokens,
            history_prompt=history_prompt,
            temp=waveform_temp,
            silent=silent,
        )
        return out


    def text_to_semantic(
        text: str,
        history_prompt: Optional[Union[Dict, str]] = None,
        temp: float = 0.7,
        silent: bool = False,
    ):
        """Generate semantic array from text.

        Args:
            text: text to be turned into audio
            history_prompt: history choice for audio cloning
            temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            silent: disable progress bar

        Returns:
            numpy semantic array to be fed into `semantic_to_waveform`
        """
        x_semantic = generate_text_semantic(
            text,
            history_prompt=history_prompt,
            temp=temp,
            silent=silent,
        )
        return x_semantic

    def semantic_to_waveform(
        semantic_tokens: np.ndarray,
        history_prompt: Optional[Union[Dict, str]] = None,
        temp: float = 0.7,
        silent: bool = False,
    ):
        """Generate audio array from semantic input.

        Args:
            semantic_tokens: semantic token output from `text_to_semantic`
            history_prompt: history choice for audio cloning
            temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            silent: disable progress bar

        Returns:
            numpy audio array at sample frequency 24khz
        """
        coarse_tokens = generate_coarse(
            semantic_tokens,
            history_prompt=history_prompt,
            temp=temp,
            silent=silent,
        )
        fine_tokens = generate_fine(
            coarse_tokens,
            history_prompt=history_prompt,
            temp=0.5,
        )
        audio_arr = codec_decode(fine_tokens)
        return audio_arr

    def generate_coarse(
        x_semantic: np.ndarray,
        history_prompt: Optional[Union[Dict, str]] = None,
        temp: float = 0.7,
        top_k: int = None,
        top_p: float = None,
        silent: bool = False,
        max_coarse_history: int = 630,  # min 60 (faster), max 630 (more context)
        sliding_window_len: int = 60,
    ):
        """
        Generate coarse audio codes from semantic tokens.
        Args:
            x_semantic: semantic token output from `text_to_semantic`
            history_prompt: history prompt, will be prepened to generated if provided
            temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            top_k: top k number of probabilities for considering during generation
            top_p: top probabilities higher than p for considering during generation
            silent: disable progress bar
            max_coarse_history: threshold for cutting coarse history (minimum 60 for faster generation, maximum 630 for more context)
            sliding_window_len: size of sliding window for generation cycle
        Returns:
            numpy audio array with coarse audio codes

        """
        semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ * N_COARSE_CODEBOOKS
        max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))
        if history_prompt is not None:
            history_prompt = _load_history_prompt(history_prompt)
            x_semantic_history = history_prompt["semantic_prompt"]
            x_coarse_history = history_prompt["coarse_prompt"]
            x_coarse_history = _flatten_codebooks(x_coarse_history) + SEMANTIC_VOCAB_SIZE
            # trim histories correctly
            n_semantic_hist_provided = np.min(
                [
                    max_semantic_history,
                    len(x_semantic_history) - len(x_semantic_history) % 2,
                    int(np.floor(len(x_coarse_history) / semantic_to_coarse_ratio)),
                ]
            )
            n_coarse_hist_provided = int(
                round(n_semantic_hist_provided * semantic_to_coarse_ratio)
            )
            x_semantic_history = x_semantic_history[-n_semantic_hist_provided:].astype(
                np.int32
            )
            x_coarse_history = x_coarse_history[-n_coarse_hist_provided:].astype(np.int32)
            x_coarse_history = x_coarse_history[:-2]
        else:
            x_semantic_history = np.array([], dtype=np.int32)
            x_coarse_history = np.array([], dtype=np.int32)
        # start loop
        n_steps = int(
            round(
                np.floor(len(x_semantic) * semantic_to_coarse_ratio / N_COARSE_CODEBOOKS)
                * N_COARSE_CODEBOOKS
            )
        )
        x_semantic = np.hstack([x_semantic_history, x_semantic]).astype(np.int32)
        x_coarse = x_coarse_history.astype(np.int32)
        base_semantic_idx = len(x_semantic_history)
        x_semantic_in = x_semantic[None]
        x_coarse_in = x_coarse[None]
        n_window_steps = int(np.ceil(n_steps / sliding_window_len))
        n_step = 0
        for _ in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
            semantic_idx = base_semantic_idx + int(round(n_step / semantic_to_coarse_ratio))
            # pad from right side
            x_in = x_semantic_in[:, np.max([0, semantic_idx - max_semantic_history]) :]
            x_in = x_in[:, :256]
            x_in = F.pad(
                torch.from_numpy(x_in),
                (0, 256 - x_in.shape[-1]),
                "constant",
                COARSE_SEMANTIC_PAD_TOKEN,
            )
            x_in = torch.hstack(
                [
                    x_in,
                    torch.tensor([COARSE_INFER_TOKEN])[None],
                    torch.from_numpy(x_coarse_in[:, -max_coarse_history:]),
                ]
            ).numpy()
            kv_cache = None
            for _ in range(sliding_window_len):
                if n_step >= n_steps:
                    continue
                is_major_step = n_step % N_COARSE_CODEBOOKS == 0

                if kv_cache is not None:
                    x_input = x_in[:, [-1]]
                else:
                    x_input = x_in

                logits, kv_cache = ov_coarse_model(x_input, past_kv=kv_cache)
                logit_start_idx = (
                    SEMANTIC_VOCAB_SIZE + (1 - int(is_major_step)) * CODEBOOK_SIZE
                )
                logit_end_idx = (
                    SEMANTIC_VOCAB_SIZE + (2 - int(is_major_step)) * CODEBOOK_SIZE
                )
                relevant_logits = logits[0, 0, logit_start_idx:logit_end_idx]
                if top_p is not None:
                    sorted_indices = np.argsort(relevant_logits)[::-1]
                    sorted_logits = relevant_logits[sorted_indices]
                    cumulative_probs = np.cumsum(F.softmax(sorted_logits))
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                    sorted_indices_to_remove[0] = False
                    relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                    relevant_logits = torch.from_numpy(relevant_logits)
                if top_k is not None:
                    relevant_logits = torch.from_numpy(relevant_logits)
                    v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                    relevant_logits[relevant_logits < v[-1]] = -float("Inf")
                probs = F.softmax(torch.from_numpy(relevant_logits) / temp, dim=-1)
                item_next = torch.multinomial(probs, num_samples=1)
                item_next = item_next
                item_next += logit_start_idx
                x_coarse_in = torch.cat(
                    (torch.from_numpy(x_coarse_in), item_next[None]), dim=1
                ).numpy()
                x_in = torch.cat((torch.from_numpy(x_in), item_next[None]), dim=1).numpy()
                del logits, relevant_logits, probs, item_next
                n_step += 1
            del x_in
        del x_semantic_in
        gen_coarse_arr = x_coarse_in.squeeze()[len(x_coarse_history) :]
        del x_coarse_in
        gen_coarse_audio_arr = (
            gen_coarse_arr.reshape(-1, N_COARSE_CODEBOOKS).T - SEMANTIC_VOCAB_SIZE
        )
        for n in range(1, N_COARSE_CODEBOOKS):
            gen_coarse_audio_arr[n, :] -= n * CODEBOOK_SIZE
        return gen_coarse_audio_arr

    def generate_fine(
        x_coarse_gen: np.ndarray,
        history_prompt: Optional[Union[Dict, str]] = None,
        temp: float = 0.5,
        silent: bool = True,
    ):
        """
        Generate full audio codes from coarse audio codes.
        Args:
            x_coarse_gen: generated coarse codebooks from `generate_coarse`
            history_prompt: history prompt, will be prepended to generated
            temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            silent: disable progress bar
        Returns:
            numpy audio array with coarse audio codes

        """
        if history_prompt is not None:
            history_prompt = _load_history_prompt(history_prompt)
            x_fine_history = history_prompt["fine_prompt"]
        else:
            x_fine_history = None
        n_coarse = x_coarse_gen.shape[0]
        # make input arr
        in_arr = np.vstack(
            [
                x_coarse_gen,
                np.zeros((N_FINE_CODEBOOKS - n_coarse, x_coarse_gen.shape[1]))
                + CODEBOOK_SIZE,
            ]
        ).astype(
            np.int32
        )  # padding
        # prepend history if available (max 512)
        if x_fine_history is not None:
            x_fine_history = x_fine_history.astype(np.int32)
            in_arr = np.hstack([x_fine_history[:, -512:].astype(np.int32), in_arr])
            n_history = x_fine_history[:, -512:].shape[1]
        else:
            n_history = 0
        n_remove_from_end = 0
        # need to pad if too short (since non-causal model)
        if in_arr.shape[1] < 1024:
            n_remove_from_end = 1024 - in_arr.shape[1]
            in_arr = np.hstack(
                [
                    in_arr,
                    np.zeros((N_FINE_CODEBOOKS, n_remove_from_end), dtype=np.int32)
                    + CODEBOOK_SIZE,
                ]
            )
        n_loops = (
            np.max([0, int(np.ceil((x_coarse_gen.shape[1] - (1024 - n_history)) / 512))])
            + 1
        )
        in_arr = in_arr.T
        for n in tqdm.tqdm(range(n_loops), disable=silent):
            start_idx = np.min([n * 512, in_arr.shape[0] - 1024])
            start_fill_idx = np.min([n_history + n * 512, in_arr.shape[0] - 512])
            rel_start_fill_idx = start_fill_idx - start_idx
            in_buffer = in_arr[start_idx : start_idx + 1024, :][None]
            for nn in range(n_coarse, N_FINE_CODEBOOKS):
                logits = ov_fine_model(
                    np.array([nn]).astype(np.int64), in_buffer.astype(np.int64)
                )
                if temp is None:
                    relevant_logits = logits[0, rel_start_fill_idx:, :CODEBOOK_SIZE]
                    codebook_preds = torch.argmax(relevant_logits, -1)
                else:
                    relevant_logits = logits[0, :, :CODEBOOK_SIZE] / temp
                    probs = F.softmax(torch.from_numpy(relevant_logits), dim=-1)
                    codebook_preds = torch.hstack(
                        [
                            torch.multinomial(probs[nnn], num_samples=1)
                            for nnn in range(rel_start_fill_idx, 1024)
                        ]
                    )
                in_buffer[0, rel_start_fill_idx:, nn] = codebook_preds.numpy()
                del logits, codebook_preds
            for nn in range(n_coarse, N_FINE_CODEBOOKS):
                in_arr[
                    start_fill_idx : start_fill_idx + (1024 - rel_start_fill_idx), nn
                ] = in_buffer[0, rel_start_fill_idx:, nn]
            del in_buffer
        gen_fine_arr = in_arr.squeeze().T
        del in_arr
        gen_fine_arr = gen_fine_arr[:, n_history:]
        if n_remove_from_end > 0:
            gen_fine_arr = gen_fine_arr[:, :-n_remove_from_end]
        return gen_fine_arr
    return generate_audio