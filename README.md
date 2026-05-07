# Deep Learning Course Project: End-to-End DL vs DL-Based Feature Learning
## CAI3105/CS460 - 12th Week Project

**College**: College of Computing and Information Technology – South Valley Campus  
**Department**: Computer Science  
**Lecturer**: Prof. Nashwa El-Bendary  
**Course Code**: CAI3105/CS460  
**Course**: Deep Learning  
**Total Marks**: 20 Marks  
**Submission Deadline**: Friday, May 8th 2026 (11:55PM)  
**Submission Platform**: Moodle LMS

---

## Project Objective

This project evaluates the performance of **End-to-End Deep Learning (DL) classification** against **DL-based Feature Learning**. The project requires a comparative study between:

1. **Approach-1**: A hybrid pipeline that utilizes a pre-trained DL model for feature extraction followed by a traditional Machine Learning (ML) classifier
2. **Approach-2**: A fully integrated end-to-end deep learning model using the same pre-trained DL model

---

## Team Structure and Submission Instructions

This project is designed for collaborative work.

### Team Requirements
- **Maximum Team Size**: Each project team must not exceed **Four (4) students**

### Submission Requirements

#### 1. PDF Report (Individual Submission)
- Each team member must submit one complete **report/documentation as a PDF file** on Moodle LMS
- The report must be formally structured according to Requirements 1, 2, 3, and 4
- All technical and analytical components must be thoroughly addressed

#### 2. Code Repository (GitHub Link)
- Each team member must submit via Moodle LMS: a link to the **public GitHub repository**
- Repository must contain the complete, runnable implementation
- Must include all scripts for:
  - Data preprocessing
  - Deep Learning (DL) feature extraction
  - Machine Learning (ML) classification

#### 3. Individual Video Presentation
- Each team member must independently record a **concise video** (~1 minute)
- Upload to personal YouTube channel and submit link via Moodle LMS
- Presentation must cover:
  - **Introduction**: Brief overview of the selected image dataset and pre-trained CNN architecture utilized
  - **Comparative Analysis**: Summary of primary results comparing End-to-End DL model against ML classifier powered by learned features
  - **Technical Insights**: Personal evaluation regarding effectiveness of pre-trained models as fixed feature extractors versus fully integrated end-to-end training

#### 4. In-Class Presentation & Discussion
- **Date & Time**: Saturday, May 9th, 2026, during scheduled lecture time
- **Presentation Duration**: 7–10 minutes per team
- **Requirements**:
  - **(1 mark) Dress Code**: Students must adhere to formal or semi-formal dress code
  - **(1 mark) A3 Poster**: Each team must generate printed A3 poster according to provided template
  - **Technical Readiness**: Teams must have laptop ready with dataset and runnable code prepared for technical discussion

---

# REQUIREMENT 1: Dataset Selection and Technical Specifications (5 marks)

Teams must select **one** publicly available image dataset from the following suggested benchmarks:

| Dataset | Problem Domain | Access Link |
|---------|---|---|
| Chest X-Ray (Pneumonia) | Medical Diagnosis | Kaggle Dataset / Mendeley Data |
| Plant Village (Leaf Disease) | Agriculture | Kaggle Dataset / UCI Repository |
| CIFAR-10 | General Object Recognition | Kaggle Dataset / TensorFlow Datasets |
| Brain MRI (Tumor Detection) | Medical Imaging | Kaggle Dataset / Mendeley Data |

For the selected dataset, provide the following:

### 1.1 Dataset Metadata (1 mark)
- **Source**: Where the dataset is obtained from
- **Problem Domain**: Type of classification problem
- **Total Number of Samples (N)**: Total number of images in the dataset

### 1.2 Technical Specifications (1 mark)
- **Image Resolution**: e.g., 224 x 224
- **Color Channels**: RGB or Grayscale
- **Total Number of Classes**: Number of classification categories

### 1.3 Data Preprocessing (1 mark)
Detail the steps taken to prepare the data, including:
- Image resizing procedures
- Pixel normalization techniques

### 1.4 Data Augmentation (1 mark)
Describe any Data Augmentation techniques employed (e.g., rotation, flipping, cropping, etc.)

**Requirements**:
- Specify which augmentation techniques are used
- Provide clear justification for selection of these techniques
- Explain necessity for the chosen dataset

### 1.5 Data Splitting (1 mark)
Specify the utilized partition of data into Training, Validation, and Testing sets

**Example**: 70% Training / 15% Validation / 15% Testing sets

---

# REQUIREMENT 2: DL Model Selection (4 marks)

### 2.1 Pre-trained Model Selection (1 mark)

Select **one** pre-trained Convolutional Neural Network (CNN) architecture from:
- ResNet50
- VGG16
- EfficientNet-B0
- MobileNetV1

This selected model will serve as the foundation for both comparative approaches:

- **Approach-1**: The pre-trained model is used as a fixed feature extractor to generate input feature vectors for a traditional ML classifier (SVM or Logistic Regression)
- **Approach-2 (End-to-End)**: The same pre-trained model is used as a complete, integrated deep learning classifier

### 2.2 Technical Justification (2 marks)

Provide a technical justification for the selected pre-trained DL model:

**Must Include**:
- Diagram or description of the model's architecture
- Architecture sourced from literature
- Original reference properly cited

### 2.3 Hyperparameter Configuration (1 mark)

Provide a detailed table listing the hyperparameters used for the DL and ML models selected for both implementations:

**Include**:
- Learning rate
- Batch size
- Number of epochs
- Optimizer
- SVM kernel type (if applicable)
- Any other relevant hyperparameters

---

# REQUIREMENT 3: Implementation Framework (5 marks)

## Approach-1: DL-Based Feature Learning + ML Classifier (3 marks)

### Implementation Steps:

1. **Feature Extraction**
   - Utilize the selected pre-trained model as a feature extractor
   - Convert images into numerical feature vectors

2. **ML Classifier Training**
   - Input learned features into a traditional ML algorithm:
     - Support Vector Machine (SVM) with linear kernel, OR
     - Logistic Regression (LR)

3. **Performance Reporting**
   - Report performance measures:
     - Accuracy
     - Precision
     - Recall
     - F-measure (F1-Score)
     - Confusion Matrix

---

## Approach-2: End-to-End Deep Learning Model (2 marks)

### Implementation Steps:

1. **Model Implementation**
   - Implement the same selected pre-trained CNN architecture
   - Use as a complete end-to-end classification model

2. **Performance Reporting**
   - Report performance measures:
     - Accuracy
     - Precision
     - Recall
     - F-measure (F1-Score)
     - Confusion Matrix

---

# REQUIREMENT 4: Comparative Analysis and Insights (4 marks)

## 4.1 Performance Comparison (2 marks)

Conduct a **comparative analysis** presenting results from:
- Approach-1 (ML classifier using DL-based learned features)
- Approach-2 (End-to-End DL model)

**Requirements**:
- Comparison must be supported by reported performance measures for both implementations
- **Highly Recommended**: Utilize infographics such as:
  - Bar charts for metric comparisons
  - Learning curves (loss/accuracy)
  - Confusion matrices
  - Any visual aids to clearly illustrate comparative analysis between approaches

---

## 4.2 Conclusion and Insights (2 marks)

Write a **concise conclusion** discussing the experimental results by addressing the following guidance questions:

### i. Performance Comparison
**How did the performance of the traditional ML classifier (based on DL-based learned features) compare to the performance of the end-to-end deep learning model?**

### ii. Advantages and Limitations
**What are the observed advantages or limitations of using pre-trained DL models as fixed feature extractors for traditional ML algorithms versus allowing the model to learn and classify in a single integrated pipeline?**

### iii. Training Efficiency
**Which approach proved to be more efficient in terms of training time?**

### iv. Recommendation for Different Environments
**Based on your findings, which strategy would you recommend for:**
- **Resource-constrained environment** (e.g., mobile or edge computing)
- **High-performance environment** (e.g., data center or cloud)

---

# BONUS MARKS (+2 Marks)

Students may earn up to **2 bonus marks** in addition to the original 20-mark total for the CAI3105-CS460-DL-12th week project by completing the following:

## Option 1: Extra CNN Model (1 Mark)
- Apply an **additional CNN architecture** using the **same dataset**
- Implement both Approach-1 and Approach-2 with this additional model
- Present comparative analysis of performance results

## Option 2: Extra Dataset (1 Mark)
- Evaluate the **primary Deep Learning model** on a **second dataset**
- Implement both approaches on the new dataset
- Present comparative analysis of performance results

**Note**: In both cases, a comparative analysis of the performance results must be presented in the final submission.

---

## Project Evaluation Summary

### Marks Breakdown

| Component | Marks |
|-----------|-------|
| Requirement 1: Dataset Selection & Technical Specs | 5 |
| Requirement 2: DL Model Selection | 4 |
| Requirement 3: Implementation Framework | 5 |
| Requirement 4: Comparative Analysis & Insights | 4 |
| In-Class Presentation: Dress Code | 1 |
| In-Class Presentation: A3 Poster | 1 |
| **Total Base Marks** | **20** |
| **Bonus Marks (Optional)** | **+2** |

---

## Submission Checklist

- [ ] PDF Report (Requirement 1-4 comprehensive)
- [ ] GitHub Repository Link (public, runnable code)
- [ ] 1-Minute Video Presentation (YouTube link)
- [ ] A3 Poster (printed for presentation)
- [ ] Formal/Semi-formal Dress Code (on presentation day)
- [ ] Laptop Ready (with dataset and runnable code)
- [ ] All Performance Metrics (Accuracy, Precision, Recall, F-measure, Confusion Matrix)
- [ ] Comparative Visualizations (charts, curves, matrices)
- [ ] Answers to All 4 Guidance Questions
- [ ] Bonus Materials (if applicable)

---

## Important Dates

- **Submission Deadline**: Friday, May 8th 2026 (11:55PM)
- **In-Class Presentation**: Saturday, May 9th 2026
- **Presentation Duration**: 7–10 minutes per team
- **Platform**: Moodle LMS

---

**Good luck with your project!**

For any questions regarding project requirements, contact your instructor.
