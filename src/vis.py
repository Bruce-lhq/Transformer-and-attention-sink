from attention_sink_module import AttentionSinkExperiment
import warnings
import os
import torch
import torch.nn.functional as F
if __name__ == "__main__":
    # 如果之前已经训练过模型并保存了检查点，就从检查点继续训练，否则从头开始训练
    if os.path.exists("attention_sink_checkpoint.pth"):
        load_from = "attention_sink_checkpoint.pth"
    else:
        load_from = None
    # 创建实验对象，训练模型，并可视化注意力图和生成文本
    corpus = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"()[]{}<>\n\\<pad>"  # 你可以根据需要调整这个字符集
    experiment = AttentionSinkExperiment(num_blocks=6, corpus=corpus, load_from=load_from,learning_rate=1e-4, sink_size=4, window_size=30, log_dir=None)
    test_text = "Hello! It is a sunny day today. I am writing my diary for today. I want you to help"
    # experiment.visualize_attention(test_text, layer_idx=-1, head_idx='mean')
    print("Generated text:", experiment.generate(test_text, max_new_tokens=10))
