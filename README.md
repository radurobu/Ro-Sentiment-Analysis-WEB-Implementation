# Ro-Sentiment-Analysis-WEB-Implementation
# This is a Flask WEB implementation of a Romanian Sentiment Analysis Model
# The model cand be seen in action at: https://roburadu.pythonanywhere.com/?

The model used to classify Romanian comments is based on the current state of the art model architecture for Natural Language processing, the BERT model. The unannotated texts used to pre-train "Romanian BERT" are drawn from three publicly available corpora: OPUS, OSCAR and Wikipedia.

The pre-trained "Romanian BERT" was sourced from Stefan Daniel Dumitrescu, Andrei-Marius Avram and Sampo Pyysalo marvelous work. The actual model can be imported directly from the Hugginface platform: https://huggingface.co/dumitrescustefan/bert-base-romanian-uncased-v1. Also check their published paper on the model training and architecture: https://arxiv.org/abs/2009.08712.

Making use of transfer-learning I further trained a new model based on Artificial Neural Network architecture to classify Romanian comments/reviews in 3 groups ("Positive", "Negative", and "Neuter"). The multiclass-classification model was trained using labeled comments scrapped from E-mag (product reviews), Facebook (financial sector comments) and CineMagia (movie reviews). Many thank to my colleagues that helped on labeling the Facebook comments!

The final model was implemented in this WEB application using Flask and SQLite modules using a simple HTML template example.
