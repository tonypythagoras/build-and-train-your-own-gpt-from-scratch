Build And Train a Modern LLM(GPT-2) From Scratch.

          Table Of Contents

1. Introduction
2. Data Preparation
3. Model Architecture
4. Coding The Model
5. Pretraining The Model
6. Post-training The Model
7. Evaluation And Validation


          1.Introduction
The four most important ingredients for modern LLM are: Data,Compute,Power and Transformer architecture.
          
Data is the text,code,images or audio, that an LLM Model reads to learn how language works. It is the examples we give a language model so it can learn patterns in language.

Compute is the raw computing power used to train an LLM model: the number of calculations the hardware has to do while the model learns.It is provided by GPUs(like NVIDIA H100 or A100), TPUs(Google custom chips) and clusters which are collections of GPUs and TPUs. Compute is measured in Flops(Floating point operations):the total number of calculations, or GPU hours(how many GPUs ran for how many hours).

Power is the electricity needed to run and cool all the compute hardwares that train and run LLM model. For instance, a modern GPU like H100 draws up to 700 watts. Watts is how much power is drawn at any moment.

The Transformer Architecture, introduced in Google's 2017 paper titled "Attention Is All You Need", made the modern LLM possible.It removed the two biggest bottlenecks that kept earlier models small and weak.That bottlenecks are: slow sequential processing and poor memory of long text.

                    STAGE I
          2. Data Preparation
Data is the experience, the culture, the language and school of modern LLMs(Large Language Models).

The first step for dataset preparation for an LLM training is to collect data from various sources including domain specific sources. The sources can be in any format such as web pages,s3 stores, sql, pdf documents, domain specific documents,curated instruction-answer pairs, human generated conversations, etc.

Notable sources for dataset preparation includes:
   1. Webtex2: web crawled data 
   2. Common crawl: highly filtered web crawl data
   3. Books1: Internet-based book corpus
   4. Books2: Internet-based book corpus
   5. Wikipedia: Internet-based collection of high-quality text
   6. Arxiv research papers
   7. TinyStories dataset: This was introduced in the research paper "TinyStories: How Small can Language Model be and Still Speaks Coherent English"
   8. FineWeb-Edu: This is a web-text filtered by an educational-quality classifier.
   9. Dolly 5k: A relatively small dataset of instruction-and-answer  examples used for teaching a model to answer questions, summarize text and follow instructions
   10. Open Assistant OASST1: Human-generated conversation and        assistant responses including multilingual examples
   11. Stanford Alpaca: A collection of instruction-response pairs used for supervised finetuning        

Training an LLM from scratch using large dataset is very expensive. For instance, the GTP-3 pretraning cost is estimated to be $4.6 million in cloud compute alone. 

Due to cost limitations and to keep things simple, we will train our LLM on a curated small dataset we can comfortably use on free Google colab, or on our consumer devices like laptop.For this purpose, we will use "TinyStories" as our pretraining dataset. It was introduced in the research paper "TinyStories: How Small Language Models Can Be and Still Speak Coherent English?". It is a dataset of short simple stories.

For post-training and finetuning, we will use "Stanford Alpaca" dataset.

An alternative to "TinyStories" used for pretraning is "FineWebEdu". This is a web text filtered by an educational-quality classifier. Models trained on it beat the ones trained on much larger unfiltered corpus on reasoning and knowledge bench marks.But training your model on this dataset is more expensive: requires more compute, storage and power. So, we will stick to using "TinyStories" as we assume anyone following this tutorial does so for educational purposes and for experimentation.

          Data Pre-processing
Data pre-processing is the cleaning and preparation of raw text before the model ever sees it. Raw data from the internet is messy, and pre-processing turns it into something a model can learn from efficiently.

The steps involved in data pre-processing are:
   1. Collection and extraction: pull text out of raw sources like web pages(striping unwanted parts like HTML tags, menus, adds, and scripts), PDFs, Code Repositories, keeping only the actual contents.
   2. Language Filtering: keep the languages you want and drop the rest using a language detector
   3. Quality Filtering: Remove low-quality text such as spam, gibberish, boilerplates,pages with mostly lists of links, or text with strange punctuation patterns.
   4. Deduplication: Remove exact and near-duplicate documents.             
   5. Safety and privacy cleaning: Remove or mask personal information and filter toxic or harmful information
   6. Decontamination: Remove texts that overlap with bechmark's test sets, so evaluation scores are not inflated by the model having seen the answers.            
          
Hugging Face is an open platform often described as "the Github of machine learning". It hosts datasets, preprocessed, ready for download and use by machine learning developers, researchers and practitioners.It is a place where pretraining and post-training datasets are shared, so machine learning teams don't have to search and clean data from scratch. For this tutorial, we will download all our datasets from Hugging Face.

To download "TinyStories" from Hugging Face for our pretraning use the following code snippet{}

To download "Stanford Alpaca" from Hugging Face for our post-training and finetuning  use the following code snippet {}. Then split it into training and validation datasets with the following code snippet {}.

                    STAGE II
          CODING  THE LLM ARCHITECTURE
LLM (Large Language Model) is a type of artificial (AI) model trained on large amout of datasets so that it understand and do various tasks like coding, language generation, computer vision, speech recognition, etc.

Examples of LLM include GPT(which powers ChatGPT),Llama, Qwen and others.In this tutorial, we will build an LLM which is a variant of GPT. Infact, we will rebuild GPT2, which is one of the earliest LLM built from scratch.

The modern LLM exploded in capability and popularity when researchers working at Google published a paper titled "Attention Is All You Need". The paper introduced the Transformer, a neural network architecture based on a mechanism called "Self Attention". The Transformer architecture as was originally published is shown below.{}

The architecture of GPT2 is shown below{}
The GPT2 architecture above has major components like:
   1. Masked Multi-head Attention
   2. Layer Normalization
   3. Transformer Blocks
   4. Feed Forward Network
   5. Linear Output Layer
   6. Residual Connection(the component with plus sign on it)
   7. Embedding layers(Token embedding and Positional embedding)
   8. Dropout

We will code the above major components one after another, starting with Masked Multi-Head Attention.

          Masked Multi-head Attention           
Masked Multi-Head Attention is a mechanism that allows a language model to learn relationships between tokens while preventing each token from looking at future tokens.It is an important component of GPT-style transformer.
The Attention formular is: Attention(Q,K,V)=Softmax(QKt/)V
This formular captures the heart of Attention Mechanism.
Below is the masked version of the Attention Mechanism in code:{}

          Layer Normalization
Layer Normalization is a technique used in neural networks, especially in Transformers, to keep the values flowing through the network at a stable rate. It helps make training more stable and predictable, reducing problems caused by activations becoming excessively large or small.The code for Layer Normalization is shown below{}

          Transformer Block
A Tranformer is a neural network architecture designed to process sequences of data by learning relationship between their elements using attention mechanism.The key components of a Transformer are:
   1. Layer Normalization
   2. Attention Mechanism
   3. Feed Forward Network
The code for Transformer is shown below{}

          Feed Forward Network
Feed Forward Network is an essential component of a Transformer block. It functions to further process each token independently after the Attention Mechanism has mixed information from different tokens. The code for Feed Forward Network is shown below{}

          Linear Output Layer
The Linear Output Layer converts each token's final hidden vector into a score for every word in the vocabulary.The layer calculates a score for every possible next token. The code for it is shown below{}

          Residual Connection
Residual connections add the old input to the new output. It enables us retain information should it get lost during processsing. In GPT-2 architecture, it is represented by (+). The formular for Residual connection is Y = F(x) + x

          Token Embedding
How do we convert tokens represented by integer IDs into a vector representation tha a neural network can learn from?. This is what Token Embedding solves. We do Token Embedding like shown below{}.

          Positional Embedding
This is what gives a Transformer information about the position of each token in a sequence.Without this, the model will not be able to  distinguish sequences that contain the samne tokens in different order.The Positional Embedding is shown below{}

          Dropout
Dropout is a metrhod to solve overfiting.It helps a model from learning its training examples too well and forgetting how to handle new examples. We do Dropout like shown below{}

Assembling the components together, the code for our LLM is shown below{}.

                    STAGE III                              
                    PRETRAINING


Pretraining is the process of teaching a Large Language Model the general patterns of language by training it on a large amount of text before teaching it to perform specific tasks.

After pretraining, an LLM becomes the next-token predictor.

The steps involved in LLM pretraining are:
  1. Data Tokenization
  2. Data Sampling
  3. Data Partitioning
  4. Hyperparameters and Training

                    Data Tokenization
Data Tokenization is the process of converting raw text into smaller units called tokens,which are then converted into numerical IDs, a required processing step for creating embeddings for LLMs

There are several open-source tools for dataset tokenization. Popular ones are:
  1. Tiktoken(developed by OpenAI)
  2. HuggingFace Transformer
  3. HuggingFace Tokenizers
  4. SentencePiece
They all support encoding and decoding like:
Note:We use titoken here
{import titoken}
{ids= tokenizer.encode(text)}
{text=tokenizer.decode(ids)}

                    Data Sampling
Data Sampling is the process of selecting and retrieving portions of your training data to feed into an LLM during training.Since your data might be composed of billions of tokens, it would be impratical to feed them all into an LLM during training. Instead, we divide the tokens into smaller portions called batches and feed them to the model during training.
Here, we use "Sliding Window" approach to sample our data.
We also need "Dataset" and "DataLoader" to encapsulate the data sampling.
                    Dataset
Dataset is an organized collection of data used to train, validate, or test a machine learning model. Our Dataset is derived from pytorch default Dataset:
{}

                    DataLoader
A DataLoader is a pytorch utility that helps feed data into model efficiently by grouping them into batches. Our DataLoader:{}

                    Data Partitioning
We partition the dataset into training,validation and test set:                

{}                    
  
                    HyperParameters and Training
Model hyperparameters are configuration values that an LLM developer set during or before an LLM training to control how trainingis performed. Examples of Hyperparameters include:
  1. Learning rate
  2. Number of epoch
  3. Batch size
  4. Weight decay
  5. Wramup step
  6. Gradient Clipping
  7. Optimizer
  8. Dropout
The Hyperparameter configuration and the training code:
{}


                       STAGE IV
             POST-TRAINING AND FINE-TUNING
Post-training is the process of improving a pretrained LLM so that it can follow instructions, answer questions helpfully and perform specific tasks. Post-training and Finetuning enable a model for: Better response-instruction following, Conversation, Task specialization, Safety and Alignment.
Steps involved in Post-training are:
  1. Data Tokenization
  2. Choosing Prompt style
  3. Implement Prompt Formating function
  4. Data Sampling
  5. Data Partitioning
  6. Hyperparameter And Training

                   Data Tokenization
This is the same as in the pretraning stage.

                 Choosing Prompt Style
Prompt style refers to the format or structure used to present training examples to a model. An example of a prompt style is "Alpaca prompt style template" which we use here.
An example of a formatted data entry for LLM training:
{}

            Implement Prompt Formating Function
Our Prompt Formating function:
{}
The formated input:
{}

           Data Sampling For Post-Training
           Create Dataset
Our Dataset:{}
            Create DataLoader
Our DataLoader will not be as simple as using the default pytorch DataLoader. Because our training datasets come with samples that contain sequences of unequal length. We need a custom "collate" function to support the DataLoader.
             Create Custom Collate Function
             {}

        Create DataLoader With The Custom Collate Function
Our custom DataLoader:
{}
       
                Data Partitioning
We partition the dataset into training,validation and test set:
{}

            Hyperparameter And Training
Our Hyperparameter configuration and training code:
{}            


































