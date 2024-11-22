________________________  in progress  ___________________________
# Classifying IoT Devices in Smart Environments Using Network Traffic Characteristics

## Overview
This repository contains the implementation of machine learning models to classify Internet of Things (IoT) devices based on network traffic characteristics. The work builds upon the research by A. Sivanathan et al. and aims to enhance IoT device classification by exploring multiple machine learning models, including **Random Forest**, **Gradient Boosting**, and **XGBoost**. The project leverages publicly available IoT traffic datasets and aims to improve classification accuracy and efficiency.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Models Used](#models-used)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Description
With the proliferation of IoT devices, identifying and classifying them based on their network behavior is crucial for enhancing security. Traditional methods relying on device signatures and IP addresses are prone to spoofing, prompting the use of network traffic analysis. This project aims to replicate and extend the findings of Sivanathan et al. by using **machine learning** to classify IoT devices based on features derived from network traffic data.

## Dataset
The dataset used in this project was sourced from A. Sivanathan et al., which includes network traffic traces from 28 different IoT devices. The devices include cameras, smart appliances, healthcare devices, and more. Data is provided in PCAP format and includes various IoT device characteristics such as packet size, inter-arrival times, and protocol usage.

- **Link to original dataset**: [Dataset from Sivanathan et al.](https://iotanalytics.unsw.edu.au/iottraces.html#bib18tmc)
  
## Preprocessing Pipeline
- The raw network traffic data (PCAP format) is preprocessed to extract relevant features like packet size, flow volume, and inter-arrival time.
- Missing values are handled efficiently by assigning default values where necessary.
- A **multi-step preprocessing** framework is employed to reduce computational costs and enhance feature extraction.

## Models Used
1. **Random Forest**: 
   - Used for classification based on decision trees. It handles high-dimensional data effectively and reduces overfitting by averaging multiple decision trees.
2. **Gradient Boosting**: 
   - An iterative model that corrects errors made by previous classifiers. It is highly effective at capturing complex patterns in the data.
3. **XGBoost**: 
   - Optimized for performance, it uses regularization techniques to prevent overfitting and is faster than other gradient boosting implementations.

## Results
The results show that **XGBoost** outperformed other models, achieving the highest accuracy of **96.26%** and the lowest **RRSE** of **25.77%**. Random Forest and Gradient Boosting also showed strong performance, with accuracies of **95.87%** and **95.02%**, respectively.

- **XGBoost Confusion Matrix**: [Link to results]
- **Random Forest Confusion Matrix**: [Link to results]
- **Gradient Boosting Confusion Matrix**: [Link to results]

## Installation
To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/kakumanuphanindra/Classifying_IoT_Devices_Based_on_Network_Patterns.git
