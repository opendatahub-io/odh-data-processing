# Standard
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, TypedDict, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

# Third Party
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def safe_print(rank, msg):
    """Only print from rank 0."""
    if rank == 0:
        logger.info(msg)


# Define model configuration
class ModelConfig(TypedDict):
    pooling_method: str
    normalize_embeddings: bool
    max_length: int
    default_instruction: str
    batch_size: int


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "Snowflake/snowflake-arctic-embed-l-v2.0": {
        "pooling_method": "cls",
        "normalize_embeddings": True,
        "max_length": 4096,
        "default_instruction": "Retrieve relevant passages:",
        "batch_size": 24,
    }
}


# pylint: disable=too-many-instance-attributes
@dataclass
class EncoderConfig:
    model_name: str
    model_config: ModelConfig
    device: torch.device
    num_gpus: int
    batch_size: int
    use_default_instruction: bool
    use_fp16: bool


class ArcticEmbedEncoder:
    def __init__(
        self,
        model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0",
        device: torch.device | None = None,
        use_fp16: bool = False,
        use_default_instruction: bool = True,
        model_path: str | None = None,  # Add optional model_path parameter
    ) -> None:
        """Initialize the Arctic encoder."""
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Model {model_name} not supported. Supported models: {list(MODEL_CONFIGS.keys())}"
            )

        # Use the provided device or default to CUDA
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Get device ID for logging
        self.device_id = self.device.index if hasattr(self.device, "index") else 0

        self.cfg = EncoderConfig(
            model_name=model_name,
            model_config=MODEL_CONFIGS[model_name],
            device=self.device,
            num_gpus=1,
            batch_size=MODEL_CONFIGS[model_name]["batch_size"],
            use_default_instruction=use_default_instruction,
            use_fp16=use_fp16,
        )
        
        self.model_path = model_path  # Store optional model path
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize model, trying local path first then downloading from HuggingFace."""
        # If a custom model path is provided, use it
        if self.model_path and os.path.exists(self.model_path):
            logger.info(f"Loading model from custom path: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(
                self.model_path,
                add_pooling_layer=False,
                trust_remote_code=True,
                local_files_only=True,
            )
        else:
            # Try local cache first
            home_dir = os.path.expanduser("~")
            local_model_path = os.path.join(
                home_dir, ".cache", "instructlab", "models", self.cfg.model_name
            )
            
            if os.path.exists(local_model_path):
                logger.info(f"Loading model from local cache: {local_model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                self.model = AutoModel.from_pretrained(
                    local_model_path,
                    add_pooling_layer=False,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            else:
                # Download from HuggingFace
                logger.info(
                    f"Model not found locally. Downloading from HuggingFace: {self.cfg.model_name}"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
                self.model = AutoModel.from_pretrained(
                    self.cfg.model_name,
                    add_pooling_layer=False,
                    trust_remote_code=True,
                )

        if self.cfg.use_fp16:
            self.model = self.model.half()

        self.model = self.model.to(self.cfg.device)
        logger.info(f"Model loaded on device: {self.cfg.device}")
        self.model.eval()

    def _prepare_inputs(
        self, texts: Union[str, List[str]], instruction: str = ""
    ) -> List[str]:
        """Prepare inputs with model-specific formatting."""
        if isinstance(texts, str):
            texts = [texts]

        # Ensure we always have an instruction
        if not instruction and not self.cfg.use_default_instruction:
            raise ValueError(
                "An instruction must be provided when use_default_instruction is False. "
                "Either provide an instruction or set use_default_instruction to True."
            )

        if (
            not instruction
            and self.cfg.use_default_instruction
            and self.cfg.model_config["default_instruction"]
        ):
            instruction = str(self.cfg.model_config["default_instruction"])

        if not instruction:  # catch if default_instruction is empty
            raise ValueError(
                "No instruction available. Either provide an instruction or ensure "
                "the model config has a valid default_instruction."
            )

        texts = [f"{instruction}: {text}" for text in texts]
        return texts

    @torch.no_grad()
    def encode(
        self,
        inputs: Union[str, List[str]],
        instruction: str = "",
        return_tensors: bool = True,
        show_progress: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Encode texts into embeddings."""
        input_was_string = isinstance(inputs, str)
        inputs = self._prepare_inputs(inputs, instruction)

        encodings = self.tokenizer(
            inputs,
            max_length=self.cfg.model_config["max_length"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.cfg.device)

        embeddings_list = []
        for i in tqdm(
            range(0, len(inputs), self.cfg.batch_size),
            disable=not show_progress or len(inputs) < 256,
        ):
            batch = {k: v[i : i + self.cfg.batch_size] for k, v in encodings.items()}
            outputs = self.model(**batch)
            # Take the first token embedding (CLS) and normalize it
            embeddings = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)
            embeddings_list.append(embeddings.cpu())

        embeddings = torch.cat(embeddings_list, dim=0)
        if input_was_string:
            embeddings = embeddings[0]

        return embeddings if return_tensors else embeddings.numpy()


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
