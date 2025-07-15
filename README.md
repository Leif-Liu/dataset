How to gen the dataset for specific docs
1. refer to the guide                 : https://ragflow.io/docs/dev/python_api_reference
2. 32 server conda env                : conda activate ragflow
3. import the docs with ragflow kb    : http://10.10.11.7/knowledge/dataset?id=62f9a54e5df611f0badf866671171edc
4. get all the trunks from ragflow kb : /home/liufeng/sdk-ragflow/get_chunks.py
5. import all the .md into easy-dataset as texts                    : easy-dataset
6. select all the texts and generate the questions                  : easy-dataset
7. select all the questions and generate the dataset                : /home/liufeng/sdk-ragflow/chunks_json/datasets-chunks-2.json
8. select all the datasets and export the file as customized format : /home/liufeng/sdk-ragflow/processed_dataset/training_ready.json
