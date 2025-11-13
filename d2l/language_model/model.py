from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from d2l.base.model import LanguageModel
from d2l.language_model.block import RNNsBlock, LSTMsBlock

class RNNLanguageModel(LanguageModel):
    """基于 RNNsBlock 的 RNN 语言模型"""
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int):
        """
        Args:
            vocab_size (int): 词表大小
            embedding_dim (int): 词向量维度 (即 RNNsBlock 的 num_input)
            num_hiddens (int): RNN 隐藏单元数量
            num_layers (int): RNN 层数
        """
        super().__init__(input_size, hidden_size, num_layers)
        
        # 2. RNN 核心：使用您提供的 RNNsBlock
        self.rnn_stack = RNNsBlock(
            embedding_dim=input_size,
            num_hiddens=hidden_size,
            num_layers=num_layers
        )
        
        # 3. 解码器（输出层）：将隐藏状态映射回词表
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, 
                input: torch.Tensor, 
                states: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            input (torch.Tensor): 输入的词元索引, 形状 (batch_size, time_steps)
            states (Optional[Tuple[torch.Tensor, torch.Tensor]]): 上一个时间步 (H, C) 的隐藏状态, 
                                             形状均为 (num_layers, batch_size, num_hiddens)
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - output_logits: 输出的 logits, 形状 (batch_size, time_steps, vocab_size)
                - final_states: 最后一个时间步的 (H, C) 隐藏状态, 形状均为 (num_layers, batch_size, num_hiddens)
        """
        # 如果未提供初始状态，则自动初始化
        batch_size = input.shape[0]
        if states is None:
            states = self.init_state(batch_size, input.device)
        
        # 2. 通过 RNN 核心
        # all_layer_outputs 形状: (num_layers, time_steps, batch_size, num_hiddens)
        # final_states 形状: (num_layers, batch_size, num_hiddens)
        all_layer_outputs, final_states = self.rnn_stack(self.one_hot(input), states)
        
        # 3. 获取最后一层 RNN 的输出
        # 我们只关心最后一层（顶层）的隐藏状态序列用于预测
        # last_layer_output 形状: (time_steps, batch_size, num_hiddens)
        last_layer_output = all_layer_outputs[-1]
        
        # 4. 通过解码器（输出层）
        # nn.Linear 会自动应用于最后一个维度 (num_hiddens)
        # output_logits 形状: (time_steps, time_steps, vocab_size)
        output_logits = self.decoder(last_layer_output)
        output_logits = output_logits.swapaxes(0, 1)  # 转换为 (batch_size, time_steps, vocab_size)
        return output_logits, final_states
    

class RNNLanguageModelTorch(LanguageModel):
    """基于 RNNsBlock 的 RNN 语言模型"""
    def __init__(self, 
                 input_size: int, 
                 hiddens_size: int, 
                 num_layers: int):
        """
        Args:
            vocab_size (int): 词表大小
            embedding_dim (int): 词向量维度 (即 RNNsBlock 的 num_input)
            num_hiddens (int): RNN 隐藏单元数量
            num_layers (int): RNN 层数
        """
        super().__init__(input_size, hiddens_size, num_layers)
        
        # 2. RNN 核心：使用您提供的 RNNsBlock
        self.rnn_stack = nn.RNN(
            input_size=input_size,
            hidden_size=hiddens_size,
            num_layers=num_layers
        )
        
        # 3. 解码器（输出层）：将隐藏状态映射回词表
        self.decoder = nn.Linear(hiddens_size, input_size)

    def forward(self, 
                input: torch.Tensor, 
                states: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            input (torch.Tensor): 输入的词元索引, 形状 (batch_size, time_steps)
            states (Optional[Tuple[torch.Tensor, torch.Tensor]]): 上一个时间步 (H, C) 的隐藏状态, 
                                             形状均为 (num_layers, batch_size, num_hiddens)
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - output_logits: 输出的 logits, 形状 (batch_size, time_steps, vocab_size)
                - final_states: 最后一个时间步的 (H, C) 隐藏状态, 形状均为 (num_layers, batch_size, num_hiddens)
        """
        # 如果未提供初始状态，则自动初始化
        batch_size = input.shape[0]
        if states is None:
            states = self.init_state(batch_size, input.device)
        
        # 2. 通过 RNN 核心
        # all_layer_outputs 形状: (time_steps, batch_size, num_hiddens)
        # final_states 形状: (num_layers, batch_size, num_hiddens)
        all_layer_outputs, final_states = self.rnn_stack(self.one_hot(input), states)
        
        # 4. 通过解码器（输出层）
        # nn.Linear 会自动应用于最后一个维度 (num_hiddens)
        # output_logits 形状: (time_steps, time_steps, vocab_size)
        output_logits = self.decoder(all_layer_outputs)
        output_logits = output_logits.swapaxes(0, 1)  # 转换为 (batch_size, time_steps, vocab_size)
        return output_logits, final_states

class LSTMLanguageModel(LanguageModel):
    """基于 LSTMBlock 的 LSTM 语言模型"""
    def __init__(self, 
                 input_size: int, 
                 hiddens_size: int, 
                 num_layers: int):
        """
        Args:
            vocab_size (int): 词表大小
            embedding_dim (int): 词向量维度 
            num_hiddens (int): RNN 隐藏单元数量
            num_layers (int): RNN 层数
        """
        super().__init__(input_size, hiddens_size, num_layers)
        self.rnn_stack = LSTMsBlock(
            input_size=input_size,
            hidden_size=hiddens_size,
            num_layers=num_layers
        )
        # 3. 解码器（输出层）：将隐藏状态映射回词表
        self.decoder = nn.Linear(hiddens_size, input_size)

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """LSTM 需要 (H, C) 形式的初始状态"""
        h0 = torch.zeros((self.num_layers, batch_size, self.num_hiddens), device=device)
        c0 = torch.zeros((self.num_layers, batch_size, self.num_hiddens), device=device)
        return h0, c0

    def forward(self, 
                input: torch.Tensor, 
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        Args:
            input (torch.Tensor): 输入的词元索引, 形状 (batch_size, time_steps)
            states (Optional[torch.Tensor]): 上一个时间步的隐藏状态, 
                                             形状 (num_layers, batch_size, num_hiddens)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - output_logits: 输出的 logits, 形状 (batch_size, time_steps, vocab_size)
                - final_states: 最后一个时间步的隐藏状态, 形状 (num_layers, batch_size, num_hiddens)
        """
        # 如果未提供初始状态，则自动初始化
        batch_size = input.shape[0]
        if states is None:
            states = self.init_state(batch_size, input.device)
        
        # 2. 通过 RNN 核心
        # all_layer_outputs 形状: (num_layers, time_steps, batch_size, num_hiddens)
        # final_states 形状: (num_layers, batch_size, num_hiddens)
        all_layer_outputs, final_states = self.rnn_stack(self.one_hot(input), states)
        
        # 3. 获取最后一层 RNN 的输出
        # 我们只关心最后一层（顶层）的隐藏状态序列用于预测
        # last_layer_output 形状: (time_steps, batch_size, num_hiddens)
        last_layer_output = all_layer_outputs[-1]
        
        # 4. 通过解码器（输出层）
        # nn.Linear 会自动应用于最后一个维度 (num_hiddens)
        # output_logits 形状: (time_steps, time_steps, vocab_size)
        output_logits = self.decoder(last_layer_output)
        output_logits = output_logits.swapaxes(0, 1)  # 转换为 (batch_size, time_steps, vocab_size)
        return output_logits, final_states


class LSTMLanguageModelTorch(LanguageModel):
    """基于 LSTMBlock 的 LSTM 语言模型"""
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int):
        """
        Args:
            vocab_size (int): 词表大小
            embedding_dim (int): 词向量维度 
            num_hiddens (int): RNN 隐藏单元数量
            num_layers (int): RNN 层数
        """
        super().__init__(input_size, hidden_size, num_layers)
        
        # 2. LSTM 核心：使用您提供的  LSTMsBlock
        self.rnn_stack = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        
        # 3. 解码器（输出层）：将隐藏状态映射回词表
        self.decoder = nn.Linear(hidden_size, input_size)

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """LSTM 需要 (H, C) 形式的初始状态"""
        h0 = torch.zeros((self.num_layers, batch_size, self.num_hiddens), device=device)
        c0 = torch.zeros((self.num_layers, batch_size, self.num_hiddens), device=device)
        return h0, c0

    def forward(self, 
                input: torch.Tensor, 
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        Args:
            input (torch.Tensor): 输入的词元索引, 形状 (batch_size, time_steps)
            states (Optional[torch.Tensor]): 上一个时间步的隐藏状态, 
                                             形状 (num_layers, batch_size, num_hiddens)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - output_logits: 输出的 logits, 形状 (batch_size, time_steps, vocab_size)
                - final_states: 最后一个时间步的隐藏状态, 形状 (num_layers, batch_size, num_hiddens)
        """
        # 如果未提供初始状态，则自动初始化
        batch_size = input.shape[0]
        if states is None:
            states = self.init_state(batch_size, input.device)
        
        # 2. 通过 RNN 核心
        # all_layer_outputs 形状: (time_steps, batch_size, num_hiddens)
        # final_states 形状: (num_layers, batch_size, num_hiddens)
        all_layer_outputs, final_states = self.rnn_stack(self.one_hot(input), states)
        
        # 4. 通过解码器（输出层）
        # nn.Linear 会自动应用于最后一个维度 (num_hiddens)
        # output_logits 形状: (time_steps, time_steps, vocab_size)
        output_logits = self.decoder(all_layer_outputs)
        output_logits = output_logits.swapaxes(0, 1)  # 转换为 (batch_size, time_steps, vocab_size)
        return output_logits, final_states
    

class Encoder(nn.Module, ABC):
    """编码器基类"""

    @abstractmethod
    def forward(
        self,
        source: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError


class Decoder(nn.Module, ABC):
    """解码器基类"""

    @abstractmethod
    def init_state(
        self,
        encoder_state: Any,
        encoder_outputs: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        target: torch.Tensor,
        state: Any,
        encoder_outputs: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError


class EncoderDecoderModel(nn.Module):
    """
    通用的 Encoder-Decoder 框架，实现 teacher forcing 训练与贪心解码。
    期待 target 序列已经包含起始符 <bos>，forward 中将使用 target[:, :-1] 作为输入，
    target[:, 1:] 作为训练标签。
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self._loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            source: 编码器输入，形状 (batch_size, src_len)
            target: 解码器输入，形状 (batch_size, tgt_len)，包含 <bos>
        Returns:
            torch.Tensor: logits，形状 (batch_size, tgt_len - 1, vocab_size)
        """
        source_mask = source_mask if source_mask is not None else self._padding_mask(source)
        target_mask = target_mask if target_mask is not None else self._padding_mask(target)
        decoder_input, _ = self._shift_target(target)

        encoder_outputs, encoder_state = self.encoder(source, source_mask)
        decoder_state = self.decoder.init_state(encoder_state, encoder_outputs, source_mask)
        logits, _ = self.decoder(decoder_input, decoder_state, encoder_outputs, target_mask[:, :-1])
        return logits

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """交叉熵损失，忽略 <pad>。"""
        _, target_y = self._shift_target(target)
        vocab_size = logits.size(-1)
        return self._loss_fn(
            logits.reshape(-1, vocab_size),
            target_y.reshape(-1),
        )

    def predict(
        self,
        source: torch.Tensor,
        max_new_tokens: int = 50,
    ) -> torch.Tensor:
        """
        使用贪心策略生成翻译/序列。
        返回的序列不包含 <bos>，长度不超过 max_new_tokens。
        """
        self.eval()
        with torch.no_grad():
            source_mask = self._padding_mask(source)
            encoder_outputs, encoder_state = self.encoder(source, source_mask)
            decoder_state = self.decoder.init_state(encoder_state, encoder_outputs, source_mask)

            batch_size = source.size(0)
            device = source.device
            step_input = torch.full(
                (batch_size, 1),
                self.bos_token_id,
                dtype=torch.long,
                device=device,
            )
            generated: Optional[torch.Tensor] = None
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            for _ in range(max_new_tokens):
                logits, decoder_state = self.decoder(step_input, decoder_state, encoder_outputs)
                step_logits = logits[:, -1, :]
                next_token = step_logits.argmax(dim=-1, keepdim=True)

                if self.eos_token_id is not None:
                    # keep feeding <eos> for finished sequences so states do not drift
                    eos_fill = torch.full_like(next_token, self.eos_token_id)
                    next_token = torch.where(finished.unsqueeze(1), eos_fill, next_token)

                if generated is None:
                    generated = next_token
                else:
                    generated = torch.cat([generated, next_token], dim=1)

                if self.eos_token_id is not None:
                    finished = finished | (next_token.squeeze(1) == self.eos_token_id)
                    if finished.all():
                        break

                step_input = next_token

            if generated is None:
                return torch.empty((batch_size, 0), dtype=torch.long, device=device)
            return generated

    def _shift_target(self, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if target.size(1) < 2:
            raise ValueError("target length must be at least 2 (include <bos> and one token).")
        decoder_input = target[:, :-1]
        decoder_target = target[:, 1:]
        return decoder_input, decoder_target

    def _padding_mask(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence.ne(self.pad_token_id)


class Seq2SeqEncoder(Encoder):
    """
    基于自定义 LSTMsBlock 的编码器。
    输入：(batch_size, src_len)
    输出：顶层隐藏状态序列 (batch_size, src_len, hidden_size) 以及最终 (H, C)。
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.rnn_stack = LSTMsBlock(embed_size, hidden_size, num_layers)

    def forward(
        self,
        source: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        del source_mask  # 当前实现未使用 mask
        batch_size = source.size(0)
        device = source.device
        if states is None:
            states = self._init_state(batch_size, device)
        embeddings = self.dropout(self.embedding(source)).transpose(0, 1)  # (seq_len, batch, embed)
        layer_outputs, final_states = self.rnn_stack(embeddings, states)
        top_outputs = layer_outputs[-1].transpose(0, 1)  # (batch, seq_len, hidden)
        return top_outputs, final_states

    def _init_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device)
        c0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device)
        return h0, c0


class Seq2SeqDecoder(Decoder):
    """
    基于 LSTMsBlock 的解码器，支持 teacher forcing 训练与贪心解码。
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
        tie_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.rnn_stack = LSTMsBlock(embed_size, hidden_size, num_layers)
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        if tie_embeddings:
            if hidden_size != embed_size:
                raise ValueError("hidden_size must equal embed_size when tying embeddings.")
            self.output_projection.weight = self.embedding.weight

    def init_state(
        self,
        encoder_state: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del encoder_outputs, source_mask
        return encoder_state

    def forward(
        self,
        target: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        encoder_outputs: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        del encoder_outputs, target_mask
        batch_size = target.size(0)
        if state is None:
            state = self._init_state(batch_size, target.device)
        embeddings = self.dropout(self.embedding(target)).transpose(0, 1)
        layer_outputs, state = self.rnn_stack(embeddings, state)
        top_outputs = layer_outputs[-1].transpose(0, 1)
        logits = self.output_projection(top_outputs)
        return logits, state # type: ignore

    def _init_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device)
        c0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device)
        return h0, c0
