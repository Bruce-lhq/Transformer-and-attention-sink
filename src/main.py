from attention_sink_module import AttentionSinkExperiment
import warnings
import os
import random
os.environ["HF_DATASETS_OFFLINE"] = "1"  # 强制使用本地数据集，避免下载
from datasets import load_dataset
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn.utils") # 忽略 seaborn 的中文字体警告，避免干扰输出
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager") # 忽略 matplotlib 的中文字体警告，避免干扰输出
raw_datasets = load_dataset("wikitext", "wikitext-103-v1")
# 在训练集中随机选取一些文本进行训练，确保文本长度适中，并构建字符集
idx = random.sample(range(len(raw_datasets['train'])), 100)  # 从训练集中随机选取100条文本
texts_for_train = [raw_datasets['train']['text'][i] for i in idx if len(raw_datasets['train']['text'][i].strip()) > 10]
corpus = "".join(set("".join(texts_for_train)))
corpus += "<pad>" # 确保包含你在 SimpleTokenizer 中需要的特殊 token
if __name__ == "__main__":
    # 如果之前已经训练过模型并保存了检查点，就从检查点继续训练，否则从头开始训练
    if os.path.exists("attention_sink_checkpoint.pth"):
        load_from = "attention_sink_checkpoint.pth"
    else:
        load_from = None
    
    # 创建实验对象，训练模型，并可视化注意力图和生成文本
    experiment = AttentionSinkExperiment(num_blocks=6, corpus=corpus, load_from=load_from,learning_rate=1e-4, sink_size=1, window_size=30, log_dir="runs/attention_sink_experiment")
    experiment.train(texts_for_train, save_path="attention_sink_checkpoint.pth", epochs=100, log_interval=5, epoch_interval=1, batch_size=8)
    test_text = "hello, how are you doing today?"
    experiment.visualize_attention(test_text, layer_idx=-1, head_idx='mean')
    print("Generated text:", experiment.generate(test_text, max_new_tokens=50))