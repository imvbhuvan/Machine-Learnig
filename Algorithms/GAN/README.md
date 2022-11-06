# Generative Adversarial Network

The promise of deep learning is to discover rich, hierarchical models that represent
probability distributions over the kinds of data encountered in artificial intelligence
applications, such as natural images, audio waveforms containing speech, and symbols in
natural language corpora. So far, the most striking successes in deep learning have involved
discriminative models, usually those that map a high-dimensional, rich sensory input to a
class label. These striking successes have primarily been based on the backpropagation and
dropout algorithms, using piecewise linear units which have a particularly well-behaved
gradient. Machine learning algorithms and neural networks can easily be fooled to
misclassify things by adding some amount of noise to data. After adding some amount of
noise, the chances of misclassifying the images increase. Hence the small rise that, is it
possible to implement something that neural networks can start visualizing new patterns like
sample train data. Thus, GANs were built that generate new fake results similar to the
original.

GANs are an exciting and rapidly changing field, delivering on the promise of
generative models in their ability to generate realistic examples across a range of problem
domains, most notably in image-to-image translation tasks such as translating photos of
summer to winter or day to night, and in generating photorealistic photos of objects, scenes,
and people that even humans cannot tell are fake.

The MNIST database of handwritten digits, available from this page, has a training
set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set
available from NIST. The digits have been size-normalized and centred in a fixed-size image.

The MNIST database was constructed from NIST's Special Database 3 and Special
Database 1 which contain binary images of handwritten digits. NIST originally designated
SD-3 as their training set and SD-1 as their test set. However, SD-3 is much cleaner and
easier to recognize than SD-1. The reason for this can be found on the fact that SD-3 was
collected among Census Bureau employees, while SD-1 was collected among high-school
students. Drawing sensible conclusions from learning experiments requires that the result be
independent of the choice of training set and test among the complete set of samples.
Therefore, it was necessary to build a new database by mixing NIST's datasets.
The MNIST training set is composed of 30,000 patterns from SD-3 and 30,000
patterns from SD-1. Our test set was composed of 5,000 patterns from SD-3 and 5,000
patterns from SD-1. The 60,000-pattern training set contained examples from approximately
250 writers. We made sure that the sets of writers of the training set and test set were disjoint.


Thank You

An algorithm demo by Vaishak Bhuvan M R
