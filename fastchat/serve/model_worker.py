"""
A model worker that executes the model.
"""
import argparse
import base64
import gc
import json
import os
from typing import List, Optional
import uuid

import torch
import torch.nn.functional as F
from transformers import set_seed, AutoProcessor
import uvicorn

from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from fastchat.model.model_adapter import (
    load_model,
    add_model_args,
    get_generate_stream_function,
)
from fastchat.modules.awq import AWQConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.modules.gptq import GptqConfig
from fastchat.serve.base_model_worker import BaseModelWorker, app
from fastchat.utils import (
    build_logger,
    get_context_length,
    str_to_torch_dtype,
)

worker_id = str(uuid.uuid4())[:8]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")


class ModelWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        device: str,
        num_gpus: int,
        max_gpu_memory: str,
        revision: str = None,
        dtype: Optional[torch.dtype] = None,
        load_8bit: bool = False,
        cpu_offloading: bool = False,
        gptq_config: Optional[GptqConfig] = None,
        awq_config: Optional[AWQConfig] = None,
        exllama_config: Optional[ExllamaConfig] = None,
        xft_config: Optional[XftConfig] = None,
        stream_interval: int = 2,
        conv_template: Optional[str] = None,
        embed_in_truncate: bool = False,
        seed: Optional[int] = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template=conv_template,
        )

        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
        self.model, self.tokenizer = load_model(
            model_path,
            revision=revision,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            dtype=dtype,
            load_8bit=load_8bit,
            cpu_offloading=cpu_offloading,
            gptq_config=gptq_config,
            awq_config=awq_config,
            exllama_config=exllama_config,
            xft_config=xft_config,
            debug=debug,
        )
        if 'videoscore' in model_path.lower():
            self.processor = AutoProcessor.from_pretrained(model_path,torch_dtype=torch.bfloat16)
        
        self.device = device
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_len = get_context_length(self.model.config)
        self.generate_stream_func = get_generate_stream_function(self.model, model_path)
        self.stream_interval = stream_interval
        self.embed_in_truncate = embed_in_truncate
        self.seed = seed

        if not no_register:
            self.init_heart_beat()

    def generate_stream_gate(self, params):
        if self.device == "npu":
            import torch_npu

            torch_npu.npu.set_device("npu:0")
        self.call_ct += 1

        try:
            if self.seed is not None:
                set_seed(self.seed)
            for output in self.generate_stream_func(
                self.model,
                self.tokenizer,
                params,
                self.device,
                self.context_len,
                self.stream_interval,
            ):
                ret = {
                    "text": output["text"],
                    "error_code": 0,
                }
                if "usage" in output:
                    ret["usage"] = output["usage"]
                if "finish_reason" in output:
                    ret["finish_reason"] = output["finish_reason"]
                if "logprobs" in output:
                    ret["logprobs"] = output["logprobs"]
                yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"

    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())

    def __process_embed_chunk(self, input_ids, attention_mask, **model_type_dict):
        if model_type_dict.get("is_bert"):
            model_output = self.model(input_ids)
            if model_type_dict.get("is_robert"):
                data = model_output.last_hidden_state
            else:
                data = model_output[0]
        elif model_type_dict.get("is_t5"):
            model_output = self.model(input_ids, decoder_input_ids=input_ids)
            data = model_output.encoder_last_hidden_state
        else:
            model_output = self.model(input_ids, output_hidden_states=True)
            if model_type_dict.get("is_chatglm"):
                data = model_output.hidden_states[-1].transpose(0, 1)
            else:
                data = model_output.hidden_states[-1]

        if hasattr(self.model, "use_cls_pooling") and self.model.use_cls_pooling:
            sum_embeddings = data[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
            masked_embeddings = data * mask
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
        token_num = torch.sum(attention_mask).item()

        return sum_embeddings, token_num

    def __encode_base64(self, embeddings: torch.Tensor) -> List[str]:
        embeddings = embeddings.cpu()
        return [
            base64.b64encode(e.numpy().tobytes()).decode("utf-8") for e in embeddings
        ]

    @torch.inference_mode()
    def get_videoscore_embeddings(self, params):
        import av
        import numpy as np
        from typing import List
        from PIL import Image
        import requests
        from io import BytesIO

        input_string_list = params["input"]
        assert len(input_string_list) == 1
        input_string = input_string_list[0]
        video_prompt, video_path = input_string.split("@@@")
        
    
        REGRESSION_QUERY_PROMPT = """
        Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
        please watch the following frames of a given video and see the text prompt for generating the video,
        then give scores from 5 different dimensions:
        (1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
        (2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
        (3) dynamic degree, the degree of dynamic changes
        (4) text-to-video alignment, the alignment between the text prompt and the video content
        (5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

        for each dimension, output a float number from 1.0 to 4.0,
        the higher the number is, the better the video performs in that sub-score, 
        the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video)
        Here is an output example:
        visual quality: 3.2
        temporal consistency: 2.7
        dynamic degree: 4.0
        text-to-video alignment: 2.3
        factual consistency: 1.8

        For this video, the text prompt is "{text_prompt}",
        all the frames of video are as follows:
        """
        ROUND_DIGIT=3
        MAX_NUM_FRAMES=48
        
        if video_path.startswith(('http://')):
            # Handle online video
            video_buffer = BytesIO(requests.get(video_path).content)
            container = av.open(video_buffer)
        else:
            # Handle local video
            container = av.open(video_path)
        # sample uniformly 8 frames from the video
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        if total_frames > MAX_NUM_FRAMES:
            indices = np.arange(0, total_frames, total_frames / MAX_NUM_FRAMES).astype(int)
        else:
            indices = np.arange(total_frames)

        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        frames = [Image.fromarray(x.to_ndarray(format="rgb24")) for x in frames]
        eval_prompt = REGRESSION_QUERY_PROMPT.format(text_prompt=video_prompt)
        num_image_token = eval_prompt.count("<image>")
        if num_image_token < len(frames):
            eval_prompt += "<image> " * (len(frames) - num_image_token)
        print(eval_prompt)
        flatten_images = []
        for x in [frames]:
            if isinstance(x, list):
                flatten_images.extend(x)
            else:
                flatten_images.append(x)
        flatten_images = [Image.open(x) if isinstance(x, str) else x for x in flatten_images]
        inputs = self.processor(text=eval_prompt, images=flatten_images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        ret = {"embedding": logits.tolist(), "token_num": 1}
        return ret
            

    @torch.inference_mode()
    def get_embeddings(self, params):
        self.call_ct += 1
        
        try:
            tokenizer = self.tokenizer
            ret = {"embedding": [], "token_num": 0}
            if  "mantis" in str(type(self.model)):
                return self.get_videoscore_embeddings(params)
            model_type_dict = {
                
                "is_t5_xxl": "t5" in str(type(self.model)),
                "is_llama": "llama" in str(type(self.model)),
                "is_t5": "t5" in str(type(self.model)),
                "is_chatglm": "chatglm" in str(type(self.model)),
                "is_bert": "bert" in str(type(self.model)),
                "is_robert": "robert" in str(type(self.model)),
            }
#             print(f"""
#                   {str(type(self.model))}
#                   {"T5Encoder" in str(type(self.model))}
# model_type_dict: {model_type_dict}
# embed_in_truncate: {self.embed_in_truncate}
# """)
            if self.embed_in_truncate:
                encoding = tokenizer.batch_encode_plus(
                    params["input"],
                    padding=True,
                    truncation="longest_first",
                    return_tensors="pt",
                    max_length=self.context_len,
                )
            else:
                encoding = tokenizer.batch_encode_plus(
                    params["input"], padding=True, return_tensors="pt"
                )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = input_ids != tokenizer.pad_token_id

            base64_encode = params.get("encoding_format", None)

            if self.embed_in_truncate:
                embedding, token_num = self.__process_embed_chunk(
                    input_ids, attention_mask, **model_type_dict
                )
                if (
                    not hasattr(self.model, "use_cls_pooling")
                    or not self.model.use_cls_pooling
                ):
                    embedding = embedding / token_num
                normalized_embeddings = F.normalize(embedding, p=2, dim=1)
                ret["token_num"] = token_num
            else:
                all_embeddings = []
                all_token_num = 0
                for i in range(0, input_ids.size(1), self.context_len):
                    
                    chunk_input_ids = input_ids[:, i : i + self.context_len]
                    chunk_attention_mask = attention_mask[:, i : i + self.context_len]
#                     print(f"""
# {hasattr(self.model, "use_cls_pooling")}

# chunk_input_ids: {chunk_input_ids}
# chunk_attention_mask: {chunk_attention_mask}

# """)
                    # add cls token and mask to get cls embedding
                    if (
                        hasattr(self.model, "use_cls_pooling")
                        and self.model.use_cls_pooling
                    ):
                        cls_tokens = (
                            torch.zeros(
                                (chunk_input_ids.size(0), 1),
                                dtype=chunk_input_ids.dtype,
                                device=chunk_input_ids.device,
                            )
                            + tokenizer.cls_token_id
                        )
                        chunk_input_ids = torch.cat(
                            [cls_tokens, chunk_input_ids], dim=-1
                        )
                        mask = torch.ones(
                            (chunk_attention_mask.size(0), 1),
                            dtype=chunk_attention_mask.dtype,
                            device=chunk_attention_mask.device,
                        )
                        chunk_attention_mask = torch.cat(
                            [mask, chunk_attention_mask], dim=-1
                        )
                    if model_type_dict.get("is_t5_xxl"):
                        # assert len(chunk_input_ids)==1
                        # chunk_input_ids = chunk_input_ids[0]
                        # chunk_input_ids = torch.nn.functional.pad(chunk_input_ids, (0, 800 - len(chunk_input_ids)))
                        # chunk_input_ids = chunk_input_ids.unsqueeze(0)
                        # chunk_attention_mask = (chunk_input_ids!=0).to(torch.long)
                        # chunk_attention_mask = chunk_attention_mask.to(torch.long)
                        chunk_embeddings = self.model(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask)['last_hidden_state'].detach()
                        token_num = torch.sum(chunk_attention_mask).item()
                        all_embeddings.extend(list(chunk_embeddings))
                        #all_embeddings.extend([chunk_embeddings_single[chunk_attention_mask_single] for chunk_embeddings_single, chunk_attention_mask_single in zip(chunk_embeddings, chunk_attention_mask)])
                    
                    else:
                        chunk_embeddings, token_num = self.__process_embed_chunk(
                            chunk_input_ids, chunk_attention_mask, **model_type_dict
                        )
                        if (
                            hasattr(self.model, "use_cls_pooling")
                            and self.model.use_cls_pooling
                        ):
                            all_embeddings.append(chunk_embeddings * token_num)
                        else:
                            all_embeddings.append(chunk_embeddings)
                        all_token_num += token_num


                all_embeddings_tensor = torch.stack(all_embeddings)
                if model_type_dict.get("is_t5_xxl"):
                    normalized_embeddings = all_embeddings_tensor
                
                else:
                    embedding = torch.sum(all_embeddings_tensor, dim=0) / all_token_num
                    normalized_embeddings = F.normalize(embedding, p=2, dim=1)

                ret["token_num"] = all_token_num

            if base64_encode == "base64":
                out_embeddings = self.__encode_base64(normalized_embeddings)
            else:
                out_embeddings = normalized_embeddings.tolist()
            ret["embedding"] = out_embeddings

            gc.collect()
            torch.cuda.empty_cache()
            if self.device == "xpu":
                torch.xpu.empty_cache()
            if self.device == "npu":
                torch.npu.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret


def create_model_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    add_model_args(parser)
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--embed-in-truncate", action="store_true")
    parser.add_argument(
        "--limit-worker-concurrency",
        type=int,
        default=5,
        help="Limit the model concurrency to prevent OOM.",
    )
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Overwrite the random seed for each generation.",
    )
    parser.add_argument(
        "--debug", type=bool, default=False, help="Print debugging messages"
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    gptq_config = GptqConfig(
        ckpt=args.gptq_ckpt or args.model_path,
        wbits=args.gptq_wbits,
        groupsize=args.gptq_groupsize,
        act_order=args.gptq_act_order,
    )
    awq_config = AWQConfig(
        ckpt=args.awq_ckpt or args.model_path,
        wbits=args.awq_wbits,
        groupsize=args.awq_groupsize,
    )
    if args.enable_exllama:
        exllama_config = ExllamaConfig(
            max_seq_len=args.exllama_max_seq_len,
            gpu_split=args.exllama_gpu_split,
            cache_8bit=args.exllama_cache_8bit,
        )
    else:
        exllama_config = None
    if args.enable_xft:
        xft_config = XftConfig(
            max_seq_len=args.xft_max_seq_len,
            data_type=args.xft_dtype,
        )
        if args.device != "cpu":
            print("xFasterTransformer now is only support CPUs. Reset device to CPU")
            args.device = "cpu"
    else:
        xft_config = None

    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        revision=args.revision,
        no_register=args.no_register,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        gptq_config=gptq_config,
        awq_config=awq_config,
        exllama_config=exllama_config,
        xft_config=xft_config,
        stream_interval=args.stream_interval,
        conv_template=args.conv_template,
        embed_in_truncate=args.embed_in_truncate,
        seed=args.seed,
        debug=args.debug,
    )
    return args, worker


if __name__ == "__main__":
    args, worker = create_model_worker()
    if args.ssl:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=os.environ["SSL_KEYFILE"],
            ssl_certfile=os.environ["SSL_CERTFILE"],
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
