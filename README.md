# MuChin
MuChin: A Chinese Colloquial Description Benchmark for Evaluating Language Models in the Field of Music

Open materials mentioned in paper (https://arxiv.org/pdf/2402.09871.pdf).

Our dataset has been organized within the `dataset` directory. 

The code for testing five different audio understanding models utilizing a Multilayer Perceptron (MLP) approach is located in the `aud_eval` folder; this code processes input audio and outputs tags along ten auditory perception dimensions.

The code dedicated to assessing semantic similarity has been archived within the `semantic` directory. This code receives two sets of tags and computes a similarity score as output.

Moreover, the code for evaluating lyric generation is contained in the `gen_eval` folder. It accepts inputs in the form of musical sections, abstract lyrical representations denoted by sequences such as 'ccccR', as well as complete lyrics, in order to calculate the corresponding scores.
