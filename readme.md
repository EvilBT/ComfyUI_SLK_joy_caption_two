# JoyCaptionAlpha Two for ComfyUI
[English](./readme_us.md) | 中文

Joy Caption 原作者在这：https://github.com/fpgaminer/joycaption ，非常感谢他的开源！

## Recent changes
* [2024-10-22] v0.0.8: 高级批量增加前缀字幕,后缀字幕，方便训练时批量添加触发词。
* [2024-10-16] v0.0.7: 统一模型加载精度，修复模型第二次无法切换的BUG，高级批量字幕增加重命名开关。
* [2024-10-16] v0.0.6: 高级模式增加top_p与temperature，给予更多的选择，添加更多的大模型选择，我试了一下 [John6666/Llama-3.1-8B-Lexi-Uncensored-V2-nf4](https://huggingface.co/John6666/Llama-3.1-8B-Lexi-Uncensored-V2-nf4)
效果不错，你们也可以尝试使用，另外也添加了原版的模型 [Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2](https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2)，可以自行选择
* [2024-10-15] v0.0.5: 修复批处理时图片有透明通道 RGBA 时的BUG
* [2024-10-15] v0.0.4: 添加指量处理节点：字幕保存目录为空时则保存在图片文件夹下，参考工作流可以在examples目录下查看。
* [2024-10-15] v0.0.3: 修复'cuda:0'部分出错的问题，直接设置为 'cuda'
* [2024-10-14] v0.0.2: 添加注册到Comfy Manager, 可以通过它来安装该节点。修复错误的模型选择引导，原框架是基于 `unsloth/Meta-Llama-3.1-8B-Instruct` 而不是 `Meta-Llama-3.1-8B`
* [2024-10-12] v0.0.1: 基本完成[JoyCaptionAlpha Two](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two)到ComfyUI的实现


## ComfyUI上JoyCaptionAlpha Two的实现

参考自 [Comfyui_CXH_joy_caption](https://github.com/StartHua/Comfyui_CXH_joy_caption), 以及 [JoyCaptionAlpha Two](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two)

参考工作流在examples/workflows.png中获取:
![image](./examples/workflows.png)

### 安装

使用 Comfy Manager, 节点安装搜索：`JoyCaptionAlpha Two for ComfyUI` 安装即可，或者使用下面手动安装方式也可以，另外注意查看下面的相关模型下载，特别是Joy-Caption-alpha-two 模型下载（必须手动下载）

### 依赖安装

1. 把仓库下载克隆到 custom_nodes 子文件夹下。
```
cd custom_nodes
git clone https://github.com/EvilBT/ComfyUI_SLK_joy_caption_two.git
```
2. 安装相关依赖：
```angular2html
pip install -r ComfyUI_SLK_joy_caption_two\requirements.txt
```
 
- 2.1 一定要确保相关依赖的版本都不小于requirements.txt的版本要求

3. 下载相关模型。

- 3.1 最好都是手动下载到指定目录，一定要注意路径要对得上，可以参考下面的截图

4. 重启ComfyUI。

### 相关模型下载
以下的models目录是指ComfyUI根目录下的models文件夹
#### 1. google/siglip-so400m-patch14-384:

国外：[google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)

国内：[hf/google/siglip-so400m-patch14-384](https://hf-mirror.com/google/siglip-so400m-patch14-384)

会自动下载，也可以手动下载整个仓库，并把siglip-so400m-patch14-384内的文件全部复制到`models/clip/siglip-so400m-patch14-384`
![image](./examples/clip.png)
#### 2. Llama3.1-8B-Instruct 模型下载

支持两个版本：bnb-4bit是小显存的福音，我是使用这个版本的，原版的我没有测试过，可自行测试。程序会自动下载，可自行下载。

2.1 unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit

国外：[unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit)

国内：[hf/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit](https://hf-mirror.com/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit)

把整个文件夹内的内容复制到 `models\LLM\Meta-Llama-3.1-8B-Instruct-bnb-4bit` 下

2.2 unsloth/Meta-Llama-3.1-8B-Instruct

国外：[unsloth/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct)

国内：[hf/unsloth/Meta-Llama-3.1-8B-Instruct](https://hf-mirror.com/unsloth/Meta-Llama-3.1-8B-Instruct)

把下载后的整个文件夹的内容复制到`models\LLM\Meta-Llama-3.1-8B-Instruct`下
![image](./examples/Llama3.1-8b.png)

#### 3. Joy-Caption-alpha-two 模型下载（必须手动下载）

把 [Joy-Caption-alpha-two](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/tree/main) 下的`cgrkzexw-599808`
文件夹的所有内容下载复制到`models/Joy_caption_two` 下
![image](./examples/joy_caption.png)
### 重启ComfyUI之后就可以添加使用了，具体可以参考下面的图片
![image](./examples/workflows.png)

### 其他

如果你安装了 [AIGODLIKE-ComfyUI-Translation](https://github.com/AIGODLIKE/AIGODLIKE-ComfyUI-Translation) 语言包插件，你可以复制 `translation` 文件夹下的中文翻译到对应的语言包路径下，重启就可以使用中文版的了。
把 `translation/zh-CN/Nodes/Comfyui_SLK_joy_caption_two.json` 复制到目录：`AIGODLIKE-ComfyUI-Translation\zh-CN\Nodes` 即可

有问题可以开issue问我，未完全测试，我是8G显存的环境
