# skin-lesion-segmentation-project
A  Deep Learning CNN based Segmentation project.

### Installation
- Create a Conda environment or a Virtual Environment based on your convinience.

- Open terminal in the directory where the environment is created or can use CLI from any Code Editor terminal.

- Install all the dependencies listed in requirement.txt or simply run the requirement.txt file by the command 

```
pip install requirement.txt
```

- the **requirement.txt** file will be added soon.

- Once all the requirements are installed run the **main.py** python file.

- As the structure may be not in sequence, you are advised to check your requirements and run the files accordingly.

### Datasets Used

#### Skin cancer ISIC The International Skin Imaging Collaboration 

Skin cancer is one of the most common types of cancer worldwide. Early detection and diagnosis are crucial for effective treatment and management. The International Skin Imaging Collaboration (ISIC) is an initiative designed to improve the diagnosis of skin cancer through the development of advanced imaging technologies and machine learning algorithms. A significant contribution of ISIC is the creation and maintenance of a comprehensive dataset of dermoscopic images, which serves as a valuable resource for researchers and clinicians.

The ISIC dataset is a collection of high-quality dermoscopic images of skin lesions, annotated with clinical metadata and diagnosis information. This dataset is widely used for developing and benchmarking machine learning models for automated skin cancer detection.

This set consists of 2357 images of malignant and benign oncological diseases, which were formed from The International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.

The data set contains the following diseases:

- actinic keratosis
- basal cell carcinoma
- dermatofibroma
- melanoma
- nevus
- pigmented benign keratosis-seborrheic keratosis
- squamous cell carcinoma
- vascular lesion

The Skin cancer dataset by ISIC can be found on [**Skin cancer ISIC The International Skin Imaging Collaboration**](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic/)

#### MNIST HAM10k

The HAM10000 ("Human Against Machine with 10000 training images") dataset is a significant collection of dermoscopic images designed to aid in the detection and classification of skin lesions. Developed to support research in the field of dermatology and medical image analysis, the HAM10000 dataset serves as a critical resource for training and evaluating machine learning models aimed at diagnosing various types of skin conditions, including melanoma.

The HAM10000 dataset contains 10,015 dermoscopic images of pigmented lesions, annotated with detailed metadata and diagnostic labels. The dataset is characterized by its diversity, high quality, and comprehensive annotation, making it one of the most valuable resources for researchers and clinicians working on skin lesion analysis.

HAM10k provides us with Diverse Lesion Types. The dataset includes images of seven different types of skin lesions:

- Melanocytic nevi (nv)
- Melanoma (mel)
- Benign keratosis-like lesions (bkl)
- Basal cell carcinoma (bcc)
- Actinic keratoses (akiec)
- Vascular lesions (vasc)
- Dermatofibroma (df)

The MNIST HAM10k dataset can be found on [**HAM10k**](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/)
