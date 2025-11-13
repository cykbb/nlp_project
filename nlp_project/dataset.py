import re
from typing import List, Tuple, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer
from d2l.dataset import Dataset
from datasets import load_dataset

class AGNewsWrapper(TorchDataset):
    """Wrapper class to make AG_NEWS data compatible with PyTorch DataLoader.
    
    这里存的是原始的 (text, label)，不做任何预处理。
    """

    def __init__(self, data: List[Tuple[Union[str, List[str]], int]]) -> None:
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Union[str, List[str]], int]:
        return self.data[idx]

class AGNewsTokenizedWrapper(TorchDataset):
    """使用 HuggingFace tokenizer 的 AG News 封装。
    
    __getitem__ 返回一个 dict + label：
      {
        "input_ids": ...,
        "attention_mask": ...,
        # 视模型而定，可能还有 "token_type_ids"
        "labels": ...
      }
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_length: int = 128,
        html_tag_pattern: Optional[re.Pattern] = re.compile(r"<.*?>"),
        url_pattern: Optional[re.Pattern] = re.compile(r"https?://\S+|www\.\S+"),
    ) -> None:
        assert len(texts) == len(labels)
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.html_tag_pattern = html_tag_pattern
        self.url_pattern = url_pattern

    def _normalize_text(self, text: str) -> str:
        """简单清洗一下文本（去掉 HTML tag 和 URL），然后交给 tokenizer."""
        if self.html_tag_pattern is not None:
            text = self.html_tag_pattern.sub(" ", text)
        if self.url_pattern is not None:
            text = self.url_pattern.sub(" ", text)
        return text.strip()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]

        text = self._normalize_text(text)

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # encoding 里每个 value 是形状 (1, max_length)，这里 squeeze 掉 batch 维
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

class AGNewsDataset(Dataset):
    HTML_TAG_PATTERN = re.compile(r"<.*?>")
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

    def __init__(
        self,
        root: str = "../data/AGNews",
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 128,
        tokenizer_kwargs: Optional[dict] = None,
    ) -> None:
        """AG News 数据集封装，内部同时保留：
        - 原始文本 (self.train / self.test)
        - tokenizer 处理后的数据 (self.train_processed / self.test_processed)

        参数
        ----
        root: HuggingFace datasets 的 cache_dir
        tokenizer_name: 传给 AutoTokenizer.from_pretrained 的模型名
        max_length: tokenizer 截断/填充长度
        tokenizer_kwargs: 传给 AutoTokenizer 的额外参数，例如 {"use_fast": True}
        """
        self.root = root
        self.text_labels = [
            "World",
            "Sports",
            "Business",
            "Sci/Tech",
        ]
        self.max_length = max_length

        # 1) 加载原始 AG_NEWS 数据 (text, label)
        self.train_data, self.test_data = self.generate(self.root)

        # 2) 加载 HuggingFace tokenizer
        tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, **tokenizer_kwargs
        )

        # 拆出文本和标签
        train_texts = [text for text, _ in self.train_data]
        test_texts = [text for text, _ in self.test_data]
        train_labels = [label for _, label in self.train_data]
        test_labels = [label for _, label in self.test_data]

        # 3) 创建原始文本 wrapper（不处理）
        self.train = AGNewsWrapper(self.train_data)
        self.test = AGNewsWrapper(self.test_data)

        # 4) 创建 tokenizer 版 wrapper（on-the-fly 调 tokenizer）
        self.train_tokenized = AGNewsTokenizedWrapper(
            train_texts,
            train_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            html_tag_pattern=self.HTML_TAG_PATTERN,
            url_pattern=self.URL_PATTERN,
        )
        self.test_tokenized = AGNewsTokenizedWrapper(
            test_texts,
            test_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            html_tag_pattern=self.HTML_TAG_PATTERN,
            url_pattern=self.URL_PATTERN,
        )

        self.train_size = len(self.train)
        self.test_size = len(self.test)
    
    @classmethod
    def generate(cls, root: str) -> Tuple[List, List]:
        try:
            dataset = load_dataset("ag_news", cache_dir=root)

            # (text, label)，text = title + " " + description
            train_data = [
                (item["text"], item["label"]) for item in dataset["train"]  # type: ignore
            ]
            test_data = [
                (item["text"], item["label"]) for item in dataset["test"]  # type: ignore
            ]
            return train_data, test_data
        except ImportError:
            raise ImportError(
                "Please install datasets library: pip install datasets\n"
                "The torchtext library has compatibility issues. "
                "We recommend using Hugging Face datasets instead."
            )
    
    def get_text_labels(self, labels: List[int]) -> List[str]:
        return [self.text_labels[int(i)] for i in labels]
    
    def preprocess_text(self, text: str):
        """对一条任意文本应用和训练集同样的 tokenizer 预处理，方便推理时使用。"""
        text = self.HTML_TAG_PATTERN.sub(" ", str(text))
        text = self.URL_PATTERN.sub(" ", text)
        text = text.strip()
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # 返回单条样本的 dict[tensor]，已经去掉 batch 维
        return {k: v.squeeze(0) for k, v in encoding.items()}
    
    def get_train_dataloader(
        self,
        batch_size: int = 64,
        tokenized: bool = False,
        shuffle: bool = True
    ) -> DataLoader:
        """tokenized=False: 返回 (text, label)
           tokenized=True: 返回 (encoding_dict, labels) 形式（实际上是 DataLoader 的默认 collate）
        """
        dataset = self.train_tokenized if tokenized else self.train
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
    def get_test_dataloader(
        self,
        batch_size: int = 64,
        tokenized: bool = False, 
        shuffle: bool = False
    ) -> DataLoader:
        dataset = self.test_tokenized if tokenized else self.test
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )