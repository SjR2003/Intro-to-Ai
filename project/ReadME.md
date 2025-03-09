# 🛠 This repository is under modification 🛠 


# SER 

In this project i tried to build a hybrid model including CNN and Fuzzy model to speech emotion recognition.  

## Run Locally  

Clone the project  

~~~bash  
  git clone https://github.com/SjR2003/Intro-to-Ai.git
~~~

Go to the project directory  

~~~bash  
  cd Intro-to-Ai/project
~~~

Install dependencies  

~~~bash  
npm install
~~~

Start the app  

~~~bash  
npm run start
~~~

## Datasets:

    - [Train dataset](https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1)
    - [Val dataset](https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1)
    - [test dataset](https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1)


## Workflow  

### Train steps:

        - Load train dataset
        - Load val dataset
        - Augmentation of datasets
        - Split datasets to CNN model data and Fuzzy model data
        - Feature extraction
        - Crate a CNN model
        - Train CNN model with CNN model data (train and validation)
        - Save CNN model
        - Load CNN model
        - Inference the CNN model with Fuzzy model data
        - Train Fuzzy model with CNN result and Fuzzy model data (train and validation)
        - Save Fuzzy model

### Test steps:

        -Load CNN and Fuzzy model
        -Load test dataset
        -Feature extraction
        -Inference the CNN model 
        -Inference the Fuzzy model 
        -Show results

## License  

[MIT](https://choosealicense.com/licenses/mit/)


