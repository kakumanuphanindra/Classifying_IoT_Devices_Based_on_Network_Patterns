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
- [References](#references)
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

- **XGBoost Confusion Matrix**: [[Link to results](https://github.com/kakumanuphanindra/Classifying_IoT_Devices_Based_on_Network_Patterns/tree/master/stage_1/xgboot/previous_run_outputs)]
- **Random Forest Confusion Matrix**: [[Link to results](https://github.com/kakumanuphanindra/Classifying_IoT_Devices_Based_on_Network_Patterns/tree/master/stage_1/random_forest_code/previous_run_outputs)]
- **Gradient Boosting Confusion Matrix**: [[Link to results](https://github.com/kakumanuphanindra/Classifying_IoT_Devices_Based_on_Network_Patterns/tree/master/stage_1/gb/previous_run_output)]

## Installation
To run the project locally you need python installed and follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/kakumanuphanindra/Classifying_IoT_Devices_Based_on_Network_Patterns.git
   ```
2. install dependencies
   ```bash
   numpy
   scapy
   matplotlib
   seaborn
   pandas
   sklearn
   ```
3. Download dataset from [here](https://iotanalytics.unsw.edu.au/iottraces.html#bib18tmc) and place all pcap file in ```attributes_processing/pcap_files```.
4. Execute ```attributes_processing/processing_data.py```
5. place the generated output ```all_attr_output.csv``` file in ```stage_0/bag_of_dst_ports_and_NB_1``` then execute ```bag_of_words_and_nv_dst_port```.
6. place the generated output ```output_with_dst_port.csv``` file in ```stage_0/bag_of_dns_and_NB_2``` then execute ```model_dns```.
7. place the generated output ```output_with_dns.csv``` file in ```stage_0/bag_of_cipher_and_NB_3``` then execute ```model_cipher```.
8. place the generated output ```output_with_cipher.csv``` file in ```stage_1/required model path``` and execute for expected results.

## References

1.	A. Sivanathan et al., "Classifying IoT Devices in Smart Environments Using Network Traffic Characteristics," in IEEE Transactions on Mobile Computing, vol. 18, no. 8, pp. 1745-1759, 1 Aug. 2019, doi: 10.1109/TMC.2018.2866249.
2.	[2]	A. Sivanathan et al., "Characterizing and classifying IoT traffic in smart cities and campuses," 2017 IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS), Atlanta, GA, USA, 2017, pp. 559-564, doi: 10.1109/INFCOMW.2017.8116438.
3.	S. Alexander and R. Droms, “DHCP Options and BOOTP vendor extensions,” Internet Requests for Comments, RFC Editor, RFC 2132, Mar. 1997.
4.	https://github.com/cisco/joy
5.	https://github.com/kakumanuphanindra/Classifying_IoT_Devices_Based_on_Network_Patterns

## Acknowledgements

- I acknowledge the work of **A. Sivanathan et al.**, whose research and publicly available dataset formed the foundation of this project.  
- Special thanks to the developers of **Wireshark** and **Joy**, which were essential for network traffic analysis and feature extraction.  
- I am grateful for the open-source contributions of libraries like **Scikit-learn**, **XGBoost**, and **Pandas**, which made model development and evaluation seamless.  
- Finally, I extend my gratitude to the **Dr. Yiheng Liang, Dr. Jin Lu & Dr. Gong Chen** for continues guidance and support to complete this project at **University of Georgia**.
