import transformers
import torch
import os,tqdm,json,time
from utils import *
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


#CUDA_VISIBLE_DEVICES=0 python find_prompt.py
input_str = """### Context:
Please refer to the instruction and task information, compare the standard answer and the predicted answer, provide your feedback for whether you think the predict answer is proper and the reasons. You need to follow the examples we provided.

### Instruction:
In this task we ask you to write answer to a question that involves “transient v. stationary" events, i.e., the understanding of whether an event will change over time or not. For example, the sentence "he was born in the U.S." contains a stationary event since it will last forever; however, "he is hungry" contains a transient event since it will remain true for a short period of time.  Note that a lot of the questions could have more than one correct answers. We only need a single most-likely answer. Please try to keep your "answer" as simple as possible. Concise and simple "answer" is preferred over those complex and verbose ones.   

### Examples:
### Task:
Sentence: Jack played basketball after school, after which he was very tired. 
Question: Was Jack still tired the next day?

### Standard Answer:
No.

### Predicted Answer:
No.

###Feedback:
Typically fatigue after playing in school goes away the next day.. So the answer should be No.

### Task:
Sentence: He was born in China, so he went to the Embassy to apply for a U.S. Visa. 
Question: Was he born in China by the time he gets the Visa?

### Standard Answer:
Yes.

### Predicted Answer:
Yes.

###Feedback:
"Born in China" is Stationary, i.e., it doesn't change over time.. So the answer should be Yes.

### Task:
Sentence: Some government officials were concerned that terrorists would take advantage of such breakdowns. 
Question: Will the officials be concerned once things are fixed?

### Standard Answer:
yes they will always be worried.

### Predicted Answer:
answer_fake

### Feedback:"""


tokenizer = load_tokenizer("/home/qianxi/scratch/laffi/models/7b")                     
model = load_model("/home/qianxi/scratch/laffi/models/7b", four_bit_quant=True, adapter_path=None)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens=100, 
    do_sample=True,
    # num_beams=3,
    num_return_sequences=10,
    #batch_size=1

)

res = pipeline(input_str)
# print(res)
# assert 1==2
output_text = [res[i]['generated_text'][len(input_str):].split('\n\n')[0].strip() for i in range(len(res))]
responses = output_text

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def major_vote_response(model, tokenizer, responses, contamination, batch_size):
    def find_most_centered_data(center, reduced_vectors):
        distances = np.linalg.norm(reduced_vectors - center, axis=1)
        idx = np.argmin(distances)

        return idx

    def batch_encode_strings(bert, tokenizer, strings, batch_size):
        model = bert
        model.eval()  # Set the model to evaluation mode

        # Initialize list to store the vectors
        vectors = []

        # Process strings in batches
        for i in range(0, len(strings), batch_size):
            # Prepare batch data
            batch = strings[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)

            # Move tensors to the device where the model is located
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            # Get hidden states
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state  # Use the last hidden states
            
            # Compute mean of all token embeddings for each sentence in the batch
            batch_vectors = hidden_states.mean(dim=1)
            vectors.extend(batch_vectors)

        # Convert the list of tensors to a single tensor
        vectors = torch.stack(vectors)
        return vectors

    # Encode responses to get the vectors
    response_vectors = batch_encode_strings(model, tokenizer, responses, batch_size)

    # # Convert list of tensors to a single tensor
    # response_vectors = torch.stack(encoded_responses)

    # Dimensionality reduction using PCA
    pca = PCA(n_components=2)
    response_vectors_2d = pca.fit_transform(response_vectors.numpy())

    # Save the 2D vectors as a torch tensor locally
    
    # Outlier detection
    iso_forest = IsolationForest(contamination=contamination)
    outliers = iso_forest.fit_predict(response_vectors_2d)
    valid_indices = [i for i, x in enumerate(outliers) if x == 1]
    invalid_indices = [i for i, x in enumerate(outliers) if x == -1]

    pure_vectors = response_vectors_2d[valid_indices]
    outliers = response_vectors_2d[invalid_indices]
    responses = [responses[i] for i in valid_indices]
    # Clustering
    kmeans = KMeans(n_clusters=1)
    cluster_labels = kmeans.fit_predict(pure_vectors)

    # Identify cluster center
    cluster_center = kmeans.cluster_centers_[0]

    selected_idx = find_most_centered_data(cluster_center, pure_vectors)
    data_center = pure_vectors[selected_idx]
    selected = responses[selected_idx]

    return selected, outliers, response_vectors_2d, cluster_center, data_center

    #torch.save(torch.tensor(response_vectors_2d), 'response_vectors_2d.pt')

def plotting(file_name, outliers, response_vectors_2d, cluster_center, data_center):
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(response_vectors_2d[:, 0], response_vectors_2d[:, 1], color='blue', label='Response Vectors')
    plt.scatter(cluster_center[0], cluster_center[1], color='red', label='Cluster Center')
    plt.scatter(outliers[:,0], outliers[:,1], color='yellow', label='Outliers')
    plt.scatter(data_center[0], data_center[1], color='green', label='Data Selected')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('2D PCA of Response Vectors')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name)

# Find and return the corresponding response for the cluster center
# closest, _ = torch.min(torch.sum((response_vectors - torch.tensor(cluster_center))**2, dim=1), dim=0)
# selected_response = responses[closest]
selected, outliers, response_vectors_2d, cluster_center, data_center = major_vote_response(model, tokenizer, responses, contamination=0.3, batch_size=16)
plotting("test_1.png", outliers, response_vectors_2d, cluster_center, data_center)