CONCH 🐚 
===========
## A Vision-Language Foundation Model for Computational Pathology
*Nature Medicine* <img src="conch.jpg" width="300px" align="right" />

 [Journal Link](https://www.nature.com/articles/s41591-024-02856-4) | [Open Access Read Link](https://rdcu.be/dBMf6) | [Download Model](https://huggingface.co/MahmoodLab/conch) | [Cite](#reference) 

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

## Updates
- **3/20/2025**: [One year overview of UNI & CONCH](https://www.linkedin.com/posts/faisalmmd_its-been-one-year-since-we-release-uni-and-activity-7308523636250820608-NedR?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAtTgDUBogopLVJVJOF9wEPZNmx4mbyt4OI) written by our team with updated table of research applications.
- 12/02/2024: Based on CONCH v1.5, a new SOTA multimodal slide foundation model, TITAN, has been released: [[Model]](https://huggingface.co/MahmoodLab/TITAN) [[Preprint]](https://arxiv.org/abs/2411.19666)
- 07/16/2024: Included comparisons with [Virchow](https://github.com/mahmoodlab/UNI?tab=readme-ov-file#slide-benchmarks).
- 06/15/2024: Included comparisons with [Prov-GigaPath](https://github.com/mahmoodlab/UNI?tab=readme-ov-file#slide-benchmarks).

## Research Applications using UNI & CONCH
<details>
  <summary>
    <b>Last Updated 3/20/2025</b>
  </summary>

| Paper Name   | Year | Publication  |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|------|
| [A self-supervised framework for learning whole slide representations](https://arxiv.org/abs/2402.06188)                                             | 2024 | arXiv:2402.06188                                                   |
| [Honeybee: a scalable modular framework for creating multimodal oncology datasets with foundational embedding models](https://arxiv.org/abs/2405.07460) | 2024 | arXiv:2405.07460                                                   |
| [Combining graph neural network and mamba to capture local and global tissue spatial relationships in whole slide images](https://arxiv.org/abs/2406.04377) | 2024 | arXiv:2406.04377                                                   |
| [STimage-1K4M: A histopathology image-gene expression dataset for spatial transcriptomics](https://arxiv.org/abs/2406.06393)                         | 2024 | arXiv:2406.06393                                                   |
| [Embedding-based multimodal learning on pan-squamous cell carcinomas for improved survival outcomes](https://arxiv.org/abs/2406.08521)               | 2024 | arXiv:2406.08521                                                   |
| [A clinical benchmark of public self-supervised pathology foundation models](https://arxiv.org/abs/2407.06508v1)                                     | 2024 | arXiv:2407.06508v1                                                |
| [Path-SAM2: Transfer SAM2 for digital pathology semantic segmentation](https://arxiv.org/abs/2408.03651)                                             | 2024 | arXiv:2408.03651                                                   |
| [Benchmarking foundation models as feature extractors for weakly-supervised computational pathology](https://arxiv.org/abs/2408.15823)               | 2024 | arXiv:2408.15823                                                   |
| [Pediatric brain tumor classification using digital histopathology and deep learning: evaluation of SOTA methods on a multi-center Swedish cohort](https://arxiv.org/abs/2409.01330) | 2024 | arXiv:2409.01330                                                   |
| [Evaluating Pre-trained Convolutional Neural Networks and Foundation Models as Feature Extractors for Content-based Medical Image Retrieval](https://arxiv.org/abs/2409.09430) | 2024 | arXiv:2409.09430                                                   |
| [Evaluating Deep Regression Models for WSI-Based Gene-Expression Prediction](https://arxiv.org/abs/2410.00945)                                       | 2024 | arXiv:2410.00945                                                   |
| [Deep Learning for Fetal Inflammatory Response Diagnosis in the Umbilical Cord](https://arxiv.org/abs/2411.09767)                                    | 2024 | arXiv:2411.09767                                                   |
| [Diagnostic Text-guided Representation Learning in Hierarchical Classification for Pathological Whole Slide Image](https://arxiv.org/abs/2411.10709) | 2024 | arXiv:2411.10709                                                   |
| [Leveraging Computational Pathology AI for Noninvasive Optical Imaging Analysis Without Retraining](https://arxiv.org/abs/2411.11613)                | 2024 | arXiv:2411.11613                                                   |
| [FOCUS: Knowledge-enhanced Adaptive Visual Compression for Few-shot Whole Slide Image Classification](https://arxiv.org/abs/2411.14743)             | 2024 | arXiv:2411.14743                                                   |
| [RankByGene: Gene-Guided Histopathology Representation Learning Through Cross-Modal Ranking Consistency](https://arxiv.org/abs/2411.15076)           | 2024 | arXiv:2411.15076                                                   |
| [ST-Align: A Multimodal Foundation Model for Image-Gene Alignment in Spatial Transcriptomics](https://arxiv.org/abs/2411.16793)                     | 2024 | arXiv:2411.16793                                                   |
| [Multimodal Outer Arithmetic Block Dual Fusion of Whole Slide Images and Omics Data for Precision Oncology](https://arxiv.org/abs/2411.17418)        | 2024 | arXiv:2411.17418                                                   |
| [Multimodal whole slide foundation model for pathology](https://arxiv.org/abs/2411.19666)                                                            | 2024 | arXiv:2411.19666                                                   |
| [GCUNet: A GNN-Based Contextual Learning Network for Tertiary Lymphoid Structure Semantic Segmentation in Whole Slide Image](https://arxiv.org/abs/2412.06129) | 2024 | arXiv:2412.06129                                                   |
| [A multimodal ensemble approach for clear cell renal cell carcinoma treatment outcome prediction](https://arxiv.org/abs/2412.07136)                 | 2024 | arXiv:2412.07136                                                   |
| [From Histopathology Images to Cell Clouds: Learning Slide Representations with Hierarchical Cell Transformer](https://arxiv.org/abs/2412.16715)     | 2024 | arXiv:2412.16715                                                   |
| [Vision-language models do not understand negation](https://arxiv.org/abs/2501.09425)                                                                | 2025 | arXiv:2501.09425                                                   |
| [Prior Knowledge Injection into Deep Learning Models Predicting Gene Expression from Whole Slide Images](https://arxiv.org/abs/2501.14056)          | 2025 | arXiv:2501.14056                                                   |
| [Molecular-driven Foundation Model for Oncologic Pathology](https://arxiv.org/abs/2501.16652)                                                        | 2025 | arXiv:2501.16652                                                   |
| [Dynamic Hypergraph Representation for Bone Metastasis Cancer Analysis](https://arxiv.org/abs/2501.16787)                                            | 2025 | arXiv:2501.16787                                                   |
| [Pathology Report Generation and Multimodal Representation Learning for Cutaneous Melanocytic Lesions](https://arxiv.org/abs/2502.19293)             | 2025 | arXiv:2502.19293                                                   |
| [DELST: Dual Entailment Learning for Hyperbolic Image-Gene Pretraining in Spatial Transcriptomics](https://arxiv.org/abs/2503.00804)                 | 2025 | arXiv:2503.00804                                                   |
| [Explainable Classifier for Malignant Lymphoma Subtyping via Cell Graph and Image Fusion](https://arxiv.org/abs/2503.00925)                          | 2025 | arXiv:2503.00925                                                   |
| [CrossFusion: A Multi-Scale Cross-Attention Convolutional Fusion Model for Cancer Survival Prediction](https://arxiv.org/abs/2503.02064)             | 2025 | arXiv:2503.02064                                                   |
| [Adaptive Prototype Learning for Multimodal Cancer Survival Analysis](https://arxiv.org/abs/2503.04643)                                              | 2025 | arXiv:2503.04643                                                   |
| [ecPath detects ecDNA in tumors from histopathology images](https://www.biorxiv.org/content/10.1101/2024.11.13.623494v1.abstract)                    | 2024 | bioRxiv:2024.11.13.623494v1                                        |
| [Contrastive Learning for Omics-guided Whole-slide Visual Embedding Representation](https://www.biorxiv.org/content/10.1101/2025.01.12.632280.abstract) | 2025 | bioRxiv:2025.01.12.632280                                          |
| [Multi-modal Disentanglement of Spatial Transcriptomics and Histopathology Imaging](https://www.biorxiv.org/content/10.1101/2025.02.19.638201v1)     | 2025 | bioRxiv:2025.02.19.638201v1                                       |
| [High-Parameter Spatial Multi-Omics through Histology-Anchored Integration](https://www.biorxiv.org/content/10.1101/2025.02.23.639721v1)             | 2025 | bioRxiv:2025.02.23.639721v1                                       |
| [Weakly-supervised deep learning models enable HER2-low prediction from H&E stained slides](https://breast-cancer-research.biomedcentral.com/articles/10.1186/s13058-024-01863-0) | 2024 | Breast Cancer Research                                            |
| [2DMamba: Efficient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image](https://arxiv.org/abs/2412.00678)  | 2025 | Computer Vision & Pattern Recognition (CVPR)                       |
| [Transcriptomics-guided slide representation learning in computational pathology](https://openaccess.thecvf.com/content/CVPR2024/html/Jaume_Transcriptomics-guided_Slide_Representation_Learning_in_Computational_Pathology_CVPR_2024_paper.html) | 2024 | Computer Vision & Pattern Recognition (CVPR)                       |
| [Morphological prototyping for unsupervised slide representation learning in computational pathology](https://openaccess.thecvf.com/content/CVPR2024/html/Song_Morphological_Prototyping_for_Unsupervised_Slide_Representation_Learning_in_Computational_Pathology_CVPR_2024_paper.html) | 2024 | Computer Vision & Pattern Recognition (CVPR)                       |
| [Development and validation of novel deep learning-based models for cancer histopathology image](https://openarchive.ki.se/articles/thesis/Development_and_validation_of_novel_deep_learning-_based_models_for_cancer_histopathology_image/27291567) | 2024 | Doctoral dissertation (Karolinska Institutet)                      |
| [Multistain pretraining for slide representation learning in pathology](https://eccv.ecva.net/virtual/2024/poster/429)                               | 2024 | European Conference on Computer Vision (ICCV)                      |
| [Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathology](https://openreview.net/forum?id=trj2Jq8riA) | 2025 | International Conference on Learning Representations (ICLR)        |
| [Multimodal prototyping for cancer survival prediction](https://proceedings.mlr.press/v235/song24b.html)                                            | 2024 | International Conference on Machine Learning (ICML)                |
| [High-resolution spatial transcriptomics from histology images using histosge](https://arxiv.org/abs/2407.20518)                                     | 2024 | International Conference on Bioinformatics and Biomedicine (BIBM)  |
| [Multi-resolution histopathology patch graphs for ovarian cancer subtyping](https://link.springer.com/chapter/10.1007/978-3-031-83243-7_7)           | 2024 | International Workshop on Graphs in Biomedical Image Analysis      |
| [Bridging Classification and Segmentation in Osteosarcoma Assessment via Foundation and Discrete Diffusion Models](https://arxiv.org/abs/2501.01932) | 2025 | International Symposium on Biomedical Imaging (ISBI)               |
| [1250 H&E-based cell prediction multi-classification models to capture morphologically distinct subpopulations of CD8+ T cells](https://jitc.bmj.com/content/12/Suppl_2/A1399) | 2024 | Journal for ImmunoTherapy of Cancer                                |
| [Liver fibrosis classification on trichrome histology slides using weakly supervised learning in children and young adults](https://www.sciencedirect.com/science/article/pii/S2153353924000555) | 2025 | Journal of Pathology Informatics                                   |
| [Winners of the 2024 Tuberculosis Detection Competition](https://www.linkedin.com/posts/zsoltbedohazi_winners-of-the-2024-tuberculosis-detection-activity-7186281385572065280-zpOq) | 2024 | LinkedIn post                                                      |
| [Model-based cleaning of the QUILT-1M pathology dataset for text-conditional image synthesis](https://openreview.net/forum?id=m7wYKrUjzV)             | 2024 | Medical Imaging with Deep Learning                                 |
| [Generating highly accurate pathology reports from gigapixel whole slide images with HistoGPT](https://www.medrxiv.org/content/10.1101/2024.03.15.24304211v2) | 2024 | medRxiv:2024.03.15.24304211v2                                     |
| [HIBRID: Histology and ct-DNA based Risk-stratification with Deep Learning](https://www.medrxiv.org/content/10.1101/2024.07.23.24310822.abstract)      | 2024 | medRxiv:2024.07.23.24310822                                       |
| ["SurvivMIL: A Multimodal, Multiple Instance Learning Pipeline for Survival Outcome of Neuroblastoma Patients"](https://proceedings.mlr.press/v254/naidoo24a.html) | 2024 | MICCAI Workshop on Computational Pathology with Multimodal Data (COMPAYL) |
| [Early Fusion of H&E and IHC Histology Images for Pediatric Brain Tumor Classification](https://openreview.net/forum?id=PHtzsqDi0n)                  | 2024 | MICCAI Workshop on Computational Pathology with Multimodal Data (COMPAYL) |
| [Fluoroformer: Scaling multiple instance learning to multiplexed images via attention-based channel fusion](https://arxiv.org/abs/2411.08975)        | 2024 | ML4H symposium                                                     |
| [Harnessing transcriptional regulation of alternative end-joining to predict cancer treatment](https://academic.oup.com/narcancer/article/7/1/zcaf007/8063268) | 2025 | NAR Cancer                                                         |
| [A multimodal generative AI copilot for human pathology](https://www.nature.com/articles/s41586-024-07618-3)                                          | 2024 | Nature                                                             |
| [Digital profiling of gene expression from histology images with linearized attention](https://www.nature.com/articles/s41467-024-54182-5)           | 2024 | Nature Communications                                             |
| [Demographic bias in misdiagnosis by computational pathology models](https://www.nature.com/articles/s41591-024-02885-z)                             | 2024 | Nature Medicine                                                    |
| [Hest-1k: A dataset for spatial transcriptomics and histology image analysis](https://proceedings.neurips.cc/paper_files/paper/2024/hash/60a899cc31f763be0bde781a75e04458-Abstract-Datasets_and_Benchmarks_Track.html) | 2024 | Advanced in Neural Information Processing Systems                  |
| [Rethinking Transformer for Long Contextual Histopathology Whole Slide Image Analysis](https://openreview.net/forum?id=f3oHNyqd83)                   | 2024 | Advanced in Neural Information Processing Systems                  |
| [Leveraging tumor heterogeneity: Heterogeneous graph representation learning for cancer survival prediction in whole slide images](https://proceedings.neurips.cc/paper_files/paper/2024/hash/760341adc5632de3f1cf2e8d22215a93-Abstract-Conference.html) | 2024 | Advanced in Neural Information Processing Systems                  |
| [Going Beyond H&E and Oncology: How Do Histopathology Foundation Models Perform for Multi-stain IHC and Immunology?](https://arxiv.org/abs/2410.21560) | 2024 | NeurIPS Workshop on Advancements In Medical Foundation Models      |
| [Histopathology and proteomics are synergistic for high-grade serous ovarian cancer platinum response prediction](https://www.nature.com/articles/s41698-025-00808-w) | 2025 | npj Precision Oncology                                             |
| [Deep learning for predicting prognostic consensus molecular subtypes in cervical cancer from histology images](https://www.nature.com/articles/s41698-024-00778-5) | 2025 | npj Precision Oncology                                             |
| [Integrated multicenter deep learning system for prognostic prediction in bladder cancer](https://www.nature.com/articles/s41698-024-00731-6)        | 2024 | npj Precision Oncology                                             |
| [Predicting the tumor microenvironment composition and immunotherapy response in non-small cell lung cancer from digital histopathology images](https://www.nature.com/articles/s41698-024-00765-w) | 2024 | npj Precision Oncology                                             |
| [Artificial intelligence-based morphologic classification and molecular characterization of neuroblastic tumors from digital histopathology](https://www.nature.com/articles/s41698-024-00745-0) | 2024 | npj Precision Oncology                                             |
| [Deep Learning-Enabled Integration of Histology and Transcriptomics for Tissue Spatial Profile Analysis](https://spj.science.org/doi/10.34133/research.0568) | 2025 | spj Research                                                       |
| [Validation of histopathology foundation models through whole slide image retrieval](https://www.nature.com/articles/s41598-025-88545-9)             | 2025 | Scientific Reports                                                 |
| [Deep Learning Framework for Classifying Whole-slide Multiplex Immunofluorescence Images to Predict Immunotherapy Response in Melanoma Patients](https://www.techrxiv.org/doi/full/10.36227/techrxiv.173496563.35713571) | 2024 | TechRxiv:10.36227/techrxiv.173496563.35713571                      |
| [Deep learning-based lymph node metastasis status predicts prognosis from muscle-invasive bladder cancer histopathology](https://link.springer.com/article/10.1007/s00345-025-05440-8) | 2025 | World Journal of Urology                                           |
</details>

## Preparing and loading the model
1. Request access to the model weights from the Huggingface model page [here](https://huggingface.co/MahmoodLab/conch).

2. Download the model weights 

First create the `checkpoints` directory inside the root of the repo:
```shell
mkdir -p checkpoints/conch/
```
Then download the pretrained model (`pytorch_model.bin`) and place it in the `CONCH/checkpoints/conch/` directory. 

3. Loading the model

First import the model builder:
```python
from conch.open_clip_custom import create_model_from_pretrained
```
Now you can load the model as follows (assuming you have the model weights in the `CONCH/checkpoints/conch/` directory):
```python
model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="checkpoints/conch/pytorch_model.bin")
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
See [**here**](notebooks/zeroshot_retrieval_example.ipynb) for a starter simple example.


## Additional Representative Benchmarks

A comprehensive set of benchmarks on zero-shot, few-shot classification are in the paper [[2]](https://www.nature.com/articles/s41591-024-02856-4). Some models were released after our study was in review. For a more comprehensive comparison, we have provided additional results on EBRAINS, PANDA, OncoTree, IHC ER / PR assessment, CRC-100K-Raw, and TCGA Uniform Tumor datasets as a representative set of benchmarks which cover a wide range of tissue types, diseases, difficulty levels (up to 108-classes) and staining (H&E and IHC). Results are reported using ABMIL and KNN (K=20) slide and ROI tasks respectively.

Please refer to the UNI [[1]](https://www.nature.com/articles/s41591-024-02857-3) and CONCH [[2]](https://www.nature.com/articles/s41591-024-02856-4) papers for more detailed benchmarking.

### Slide Benchmarks
| Model name     | Pretraining       |   EBRAINS-C (12 classes, Public)       |   EBRAINS-F (30 classes, Public)     |   PANDA (5 classes, Public) |   OncoTree-108 (108 classes, Internal) |   IHC ER / PR Assess. (6 classes, Internal)  |
|:---------------|:------------------|---------------------------:|-------------------------:|-----------------:|------------------:|---------------------------:|
|  |  | Balanced acc. | Balanced acc. | Quadratic-weight $\kappa$ | Balanced acc. | Quadratic-weight $\kappa$ |
| **UNI** [[1]](https://www.nature.com/articles/s41591-024-02857-3)            | Vision  |                      **0.883** |                    0.675 |            <ins>0.946</ins> |             **0.538** |     0.785 |
| **CONCH** [[2]](https://www.nature.com/articles/s41591-024-02856-4)         | Vision-language   |                      0.868 |                    **0.689** |            0.934 |             0.515 |       <ins>0.819</ins> |
| Virchow (CLS+MEAN) [[3]](https://arxiv.org/pdf/2309.07778)            | Vision  |                      0.833 |                    0.654 |            0.943 |             0.519 |     0.788 |
| Prov-GigaPath [[4]](https://www.nature.com/articles/s41586-024-07441-w)            | Vision  |                      <ins>0.875</ins> |                    <ins>0.687</ins> |            0.942 |             <ins>0.522</ins> |     **0.821** |
| Phikon [[5]](https://doi.org/10.1101/2023.07.21.23292757)         | Vision   |                      0.810  |                    0.659 |            **0.950**  |             0.486 |                      0.744 |  
| REMEDIS [[6]](https://doi.org/10.1038/s41551-023-01049-7)     | Vision   |                      0.687 |                    0.382 |            0.932 |             0.412 |                      0.762 |   
| CTransPath [[7]](https://doi.org/10.1016/j.media.2022.102559)     | Vision   |                      0.666 |                    0.514 |            0.927 |             0.399 |                      0.786 | 
| Quilt-Net [[8]](https://proceedings.neurips.cc/paper_files/paper/2023/file/775ec578876fa6812c062644964b9870-Paper-Datasets_and_Benchmarks.pdf)          | Vision-language   |                      0.728 |                    0.608 |            0.909 |             0.389 |                      0.784 | 
| PLIP [[9]](https://doi.org/10.1038/s41591-023-02504-3)           | Vision-language   |                      0.683 |                    0.562 |            0.901 |             0.369 |                      0.759 | 
| ResNet-50 (Tr) [[10]](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) | ImageNet Transfer |                      0.302 |                    0.219 |            0.831 |             0.148 |                      0.709 |

### ROI Benchmarks
| Model name     | Pretraining       |   CRC-100K-Raw (9 classes, Public)           |  TCGA Uniform Tumor (32 classes, Public)      |
|:---------------|:------------------|---------------------------:|-------------------------:|
|  |  | Balanced acc. | Balanced acc. |
| **UNI** [[1]](https://www.nature.com/articles/s41591-024-02857-3)        | Vision            |             0.925          |                **0.595** |
| **CONCH** [[2]](https://www.nature.com/articles/s41591-024-02856-4)         | Vision-language      |            **0.941**      |                    0.556 |
| Virchow (CLS+MEAN) [[3]](https://arxiv.org/pdf/2309.07778)        | Vision            |             0.919          |                0.549 |
| Virchow (CLS) [[3]](https://arxiv.org/pdf/2309.07778)        | Vision            |             0.895          |                0.544 |
| Prov-GigaPath [[4]](https://www.nature.com/articles/s41586-024-07441-w)        | Vision            |             <ins>0.929</ins>          |                <ins>0.593</ins> |
| Phikon [[5]](https://doi.org/10.1101/2023.07.21.23292757)          | Vision            |             0.845          |                    0.533 |
| REMEDIS [[6]](https://doi.org/10.1038/s41551-023-01049-7)        | Vision            |             0.908          |                    0.541 |
| CTransPath [[7]](https://doi.org/10.1016/j.media.2022.102559)     | Vision            |             0.836          |                    0.463 |
| Quilt-Net [[8]](https://proceedings.neurips.cc/paper_files/paper/2023/file/775ec578876fa6812c062644964b9870-Paper-Datasets_and_Benchmarks.pdf)     | Vision-language   |                      0.878 |                    0.359 |
| PLIP [[9]](https://doi.org/10.1038/s41591-023-02504-3)          | Vision-language   |                      0.840 |                    0.370 |
| ResNet-50  [[10]](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)    | ImageNet Transfer |                      0.797 |                    0.318 |  

## Acknowledgements

The project was built on top of amazing repositories such as [openclip](https://github.com/mlfoundations/open_clip) (used for model training),  [timm](https://github.com/huggingface/pytorch-image-models/) (ViT model implementation) and [huggingface transformers](https://github.com/huggingface/transformers) (tokenization). We thank the authors and developers for their contribution. 

## License and Terms of Use

ⓒ Mahmood Lab. This model and associated code are released under the [CC-BY-NC-ND 4.0]((https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en)) license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the CONCH model and its derivatives, which include models trained on outputs from the CONCH model or datasets created from the CONCH model, is prohibited and requires prior approval. Downloading the model requires prior registration on Hugging Face and agreeing to the terms of use. By downloading this model, you agree not to distribute, publish or reproduce a copy of the model. If another user within your organization wishes to use the CONCH model, they must register as an individual user and agree to comply with the terms of use. Users may not attempt to re-identify the deidentified data used to develop the underlying model. If you are a commercial entity, please contact the corresponding author or Mass General Brigham Innovation Office.

## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our [paper](https://www.nature.com/articles/s41591-024-02856-4):

Lu, M. Y., Chen, B., Williamson, D. F., Chen, R. J., Liang, I., Ding, T., ... & Mahmood, F. (2024). A visual-language foundation model for computational pathology. Nature Medicine.


```
@article{lu2024avisionlanguage,
  title={A visual-language foundation model for computational pathology},
  author={Lu, Ming Y and Chen, Bowen and Williamson, Drew FK and Chen, Richard J and Liang, Ivy and Ding, Tong and Jaume, Guillaume and Odintsov, Igor and Le, Long Phi and Gerber, Georg and others},
  journal={Nature Medicine},
  pages={863–874},
  volume={30},
  year={2024},
  publisher={Nature Publishing Group}
}
```

Additionally, if you find MI-Zero useful, please also consider citing the corresponding [CVPR 2023 article](https://openaccess.thecvf.com/content/CVPR2023/html/Lu_Visual_Language_Pretrained_Multiple_Instance_Zero-Shot_Transfer_for_Histopathology_Images_CVPR_2023_paper.html):

Lu, M.Y., Chen, B., Zhang, A., Williamson, D.F., Chen, R.J., Ding, T., Le, L.P., Chuang, Y.S. and Mahmood, F., 2023. Visual Language Pretrained Multiple Instance Zero-Shot Transfer for Histopathology Images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 19764-19775).
```
@InProceedings{Lu_2023_CVPR,
    author    = {Lu, Ming Y. and Chen, Bowen and Zhang, Andrew and Williamson, Drew F. K. and Chen, Richard J. and Ding, Tong and Le, Long Phi and Chuang, Yung-Sung and Mahmood, Faisal},
    title     = {Visual Language Pretrained Multiple Instance Zero-Shot Transfer for Histopathology Images},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {19764-19775}
}
```
<img src=docs/joint_logo.jpg> 
