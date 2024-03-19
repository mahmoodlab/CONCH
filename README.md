CONCH 🐚 
===========
## A Vision-Language Foundation Model for Computational Pathology
*Nature Medicine* <img src="conch.jpg" width="300px" align="right" />

 [Journal Link](https://www.nature.com/articles/s41591-024-02856-4) | [Download Model](https://huggingface.co/MahmoodLab/conch) | [Cite](#reference) 

**Abstract:** The accelerated adoption of digital pathology and advances in deep learning have enabled the development of robust models for various pathology tasks across a diverse array of diseases and patient cohorts. However, model training is often difficult due to label scarcity in the medical domain and the model's usage is limited by the specific task and disease for which it is trained. Additionally, most models in histopathology leverage only image data, a stark contrast to how humans teach each other and reason about histopathologic entities. We introduce CONtrastive learning from Captions for Histopathology (CONCH), a visual-language foundation model developed using diverse sources of histopathology images, biomedical text, and notably over 1.17 million image-caption pairs via task-agnostic pretraining. Evaluated on a suite of 14 diverse benchmarks, CONCH can be transferred to a wide range of downstream tasks involving either or both histopathology images and text, achieving state-of-the-art performance on histology image classification, segmentation, captioning, text-to-image, and image-to-text retrieval. CONCH represents a substantial leap over concurrent visual-language pretrained systems for histopathology, with the potential to directly facilitate a wide array of machine learning-based workflows requiring minimal or no further supervised fine-tuning.

## What is CONCH?
CONCH (CONtrastive learning from Captions for Histopathology) is a vision language foundation model for histopathology, pretrained on currently the largest histopathology-specific vision-language dataset of 1.17M image caption pairs. Compare to other vision language foundation models, it demonstrates state-of-the-art performance across 14 tasks in computational pathology ranging from image classification, text-to-image, and image-to-text retrieval, captioning, and tissue segmentation.
- _**Why use CONCH?**_: Compared to popular self-supervised encoders for computational pathology that were pretrained only on H&E images, CONCH may produce more performant representations for non-H&E stained images such as IHCs and special stains, and can be used for a wide range of downstream tasks involving either or both histopathology images and text. CONCH also did not use large public histology slide collections such as TCGA, PAIP, GTEX, etc. for pretraining, which are routinely used in benchmark development in computational pathology. Therefore, we make CONCH available for the research community in building and evaluating pathology AI models with minimal risk of data contamination on public benchmarks or private histopathology slide collections.

## Installation
First clone the repo and cd into the directory:
```shell
git clone https://github.com/mahmoodlab/CONCH.git
cd CONCH
```
Then create a conda env and install the dependencies:
```shell
conda create -n conch python=3.10 -y
conda activate conch
pip install --upgrade pip
pip install -e .
```

## Preparing and loading the model
1. Request access to the model weights from the Huggingface model page [here](https://huggingface.co/MahmoodLab/conch).

2. Download the model weights 
First create the `checkpoints` directory inside the root of the repo:
```shell
mkdir -p checkpoints/conch/
```
Then download the pretrained model (`conch.pt`) and place it in the `CONCH/checkpoints/conch/` directory. 

3. Loading the model
First import the model builder:
```python
from conch.open_clip_custom import create_model_from_pretrained
```
Now you can load the model as follows (assuming you have the model weights in the `CONCH/checkpoints/conch/` directory):
```python
model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="checkpoints/conch/conch.pt")
```

Alternatively, you can use the following command to download and then load the model directly from HF after requesting access:
```python
from conch.open_clip_custom import create_model_from_pretrained
model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token="<your_user_access_token>")
```

You may need to supply your huggingface user access token via `hf_auth_token=<your_token>` to `create_model_from_pretrained` for authentification. See the [HF documentation](https://huggingface.co/docs/hub/security-tokens) for more details.

Note: while the original CONCH model arechitecture also includes a multimodal decoder trained with the captioning loss of CoCa, as additional precaution to ensure that no proprietary data or Protected Health Information (PHI) is leaked untentionally, we have removed the weights for the decoder from the publicly released CONCH weights. The weights for the text encoder and the vision encoder are intact and therefore the results on all key tasks presented in the paper such as image classification and image-text retrieval are not affected. The ability of CONCH to serve as a general purpose encoder for both histopathology images and pathology-related text also remains unaffected.


## Using the model as an vision encoder for histopathology images
Given the importance of pretrained enocders currently for computational pathology tasks, we highlight that after loading the model, you can now use it to embed images as follows:
```python
from PIL import Image
image = Image.open("path_to_image.jpg")
image = preprocess(image).unsqueeze(0)
with torch.inference_mode():
    image_embs = model.encode_image(image, proj_contrast=False, normalize=False)
```
This will give you the image embeddings before the projection head and normalization, suitable for linear probe or working with WSIs under the multiple-instance learning framework. 

For image-text retrieval tasks, you should use the normalized and projected embeddings as follows:
```python
with torch.inference_mode():
    image_embs = model.encode_image(image, proj_contrast=True, normalize=True)
```

## Overview of specific usages
We provide high-level functions for loading the model and using it for inference. For model loading:
```python
from conch.open_clip_custom import create_model_from_pretrained
```
For tokenizing text:
```python
from conch.open_clip_custom import tokenize, get_tokenizer
```
For inference:
```python
from conch.downstream.zeroshot_path import zero_shot_classifier, run_mizero, run_zeroshot
```
Refer to the notebooks below for detailed examples.

### More detailed starter code for loading / using the model:
See [**here**](notebooks/basics_usage.ipynb) to get started with loading and using the model to create embeddings.

### Zeroshot classification of a image ROIs/tiles:
See [**here**](notebooks/zeroshot_classification_example_starter.ipynb) for a starter simple example.

For a full example using dataloaders and prompt ensembling see [**here**](notebooks/zeroshot_classification_example_ensemble.ipynb).

### Zeroshot classification of a WSIs using MI-Zero:
See [**here**](notebooks/MI-zeroshot_classification_example_ensemble.ipynb). Note that you will first need to tile the WSIs and convert the tiles into embeddings using the CONCH vision encoder model.

### Zeroshot cross-modality retrieval (image / text):
See [**here**](**notebooks/zeroshot_retrieval_example.ipynb**) for a starter simple example.


## Additional Representative Benchmarks

A comprehensive set of benchmarks on zero-shot, few-shot classification are in the paper [[2]](https://www.nature.com/articles/s41591-024-02856-4). Some models were released after our study was in review. For a more comprehensive comparison, we have provided additional results on EBRAINS, PANDA, OncoTree, IHC ER / PR assessment, CRC-100K-Raw, and TCGA Uniform Tumor datasets as a representative set of benchmarks which cover a wide range of tissue types, diseases, difficulty levels (up to 108-classes) and staining (H&E and IHC).

Please refer to the UNI [[1]](https://www.nature.com/articles/s41591-024-02857-3) and CONCH [[2]](https://www.nature.com/articles/s41591-024-02856-4) papers for more detailed benchmarking.

### Slide Benchmarks
| Model name     | Pretraining       |   EBRAINS-C (12 classes, Public)       |   EBRAINS-F (30 classes, Public)     |   PANDA (5 classes, Public) |   OncoTree-108 (108 classes, Internal) |   IHC ER / PR Assess. (6 classes, Internal)  |
|:---------------|:------------------|---------------------------:|-------------------------:|-----------------:|------------------:|---------------------------:|
|  |  | Balanced acc. | Quadratic-weight $\kappa$ | Balanced acc. | Balanced acc. | Quadratic-weight $\kappa$ |
| **UNI** [[1]](https://www.nature.com/articles/s41591-024-02857-3)            | Vision  |                      **0.883** |                    <ins>0.675</ins> |            <ins>0.946</ins> |             **0.538** |     0.785 |
| **CONCH** [[2]](https://www.nature.com/articles/s41591-024-02856-4)         | Vision-language   |                      <ins>0.868</ins> |                    **0.689** |            0.934 |             <ins>0.515</ins> |       **0.819** |
| Phikon [[3]](https://doi.org/10.1101/2023.07.21.23292757)         | Vision   |                      0.810  |                    0.659 |            **0.950**  |             0.486 |                      0.744 |  
| REMEDIS [[4]](https://doi.org/10.1038/s41551-023-01049-7)     | Vision   |                      0.687 |                    0.382 |            0.932 |             0.412 |                      0.762 |   
| CTransPath [[5]](https://doi.org/10.1016/j.media.2022.102559)     | Vision   |                      0.666 |                    0.514 |            0.927 |             0.399 |                      <ins>0.786<ins> | 
| Quilt-Net [[6]](https://proceedings.neurips.cc/paper_files/paper/2023/file/775ec578876fa6812c062644964b9870-Paper-Datasets_and_Benchmarks.pdf)          | Vision-language   |                      0.728 |                    0.608 |            0.909 |             0.389 |                      0.784 | 
| PLIP [[7]](https://doi.org/10.1038/s41591-023-02504-3)           | Vision-language   |                      0.683 |                    0.562 |            0.901 |             0.369 |                      0.759 | 
| ResNet-50 (Tr) [[8]](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) | ImageNet Transfer |                      0.302 |                    0.219 |            0.831 |             0.148 |                      0.709 |


### ROI Benchmarks
| Model name     | Pretraining       |   CRC-100K-Raw (9 classes, Public)           |  TCGA Uniform Tumor (32 classes, Public)      |
|:---------------|:------------------|---------------------------:|-------------------------:|
|  |  | Balanced acc. | Balanced acc. |
| **UNI** [[1]](https://www.nature.com/articles/s41591-024-02857-3)        | Vision            |             <ins>0.925</ins>          |                **0.595** |
| **CONCH** [[2]](https://www.nature.com/articles/s41591-024-02856-4)         | Vision-language      |            **0.941**      |                    <ins>0.556</ins> |
| Phikon [[3]](https://doi.org/10.1101/2023.07.21.23292757)          | Vision            |             0.845          |                    0.533 |
| REMEDIS [[4]](https://doi.org/10.1038/s41551-023-01049-7)        | Vision            |             0.908          |                    0.541 |
| CTransPath [[5]](https://doi.org/10.1016/j.media.2022.102559)     | Vision            |             0.836          |                    0.463 |
| Quilt-Net [[6]](https://proceedings.neurips.cc/paper_files/paper/2023/file/775ec578876fa6812c062644964b9870-Paper-Datasets_and_Benchmarks.pdf)     | Vision-language   |                      0.878 |                    0.359 |
| PLIP [[7]](https://doi.org/10.1038/s41591-023-02504-3)          | Vision-language   |                      0.840 |                    0.370 |
| ResNet-50  [[8]](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)    | ImageNet Transfer |                      0.797 |                    0.318 | 

## Acknowledgements

The project was built on top of amazing repositories such as [openclip](https://github.com/mlfoundations/open_clip) (used for model training),  [timm](https://github.com/huggingface/pytorch-image-models/) (ViT model implementation) and [huggingface transformers](https://github.com/huggingface/transformers) (tokenization). We thank the authors and developers for their contribution. 

## License and Terms of Use

ⓒ Mahmood Lab. This model and associated code are released under the [CC-BY-NC-ND 4.0]((https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en)) license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the CONCH model and its derivatives, which include models trained on outputs from the CONCH model or datasets created from the CONCH model, is prohibited and requires prior approval. Downloading the model requires prior registration on Hugging Face and agreeing to the terms of use. By downloading this model, you agree not to distribute, publish or reproduce a copy of the model. If another user within your organization wishes to use the CONCH model, they must register as an individual user and agree to comply with the terms of use. Users may not attempt to re-identify the deidentified data used to develop the underlying model. If you are a commercial entity, please contact the corresponding author or Mass General Brigham Innovation Office.

## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our [paper](https://www.nature.com/articles/s41591-024-02856-4):

Lu, M. Y., Chen, B., Williamson, D. F., Chen, R. J., Liang, I., Ding, T., ... & Mahmood, F. (2024). A visual-language foundation model for computational pathology. Nature Medicine.


```
@article{lu2024avisionlanguage,
  title={A Vision Language Foundation Model for Computational Pathology},
  author={Lu, Ming Y and Chen, Bowen and Williamson, Drew FK and Chen, Richard J and Liang, Ivy and Ding, Tong and Jaume, Guillaume and Odintsov, Igor and Zhang, Andrew and Le, Long Phi and others},
  journal={Nature Medicine},
  publisher={Nature Publishing Group}
}
```
<img src=docs/joint_logo.jpg> 
