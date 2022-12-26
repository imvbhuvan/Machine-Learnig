## Naive Bayes Algorithm for Spam Detection

The Naive Bayes algorithm is a machine learning technique that is commonly used in spam detection. It is based on the idea that given a set of features (such as the presence of certain words or phrases in an email), we can predict the probability of an email being spam.

To classify an email as spam or not, the Naive Bayes algorithm uses Bayes' theorem, which states that the probability of an event (in this case, the email being spam) occurring given certain evidence (the presence of certain features) is equal to the probability of the evidence occurring given that the event has occurred, multiplied by the probability of the event occurring in the first place.

The Naive Bayes algorithm makes the assumption that the features in the email are independent of one another, which is why it is called "naive." This assumption allows the algorithm to make predictions more quickly, but it may not always be accurate.

The model built here achieves around 98% accuray for the given dataset.
