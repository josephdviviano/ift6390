
CNN material that you should know:
- Local connectivity
- Parameter sharing
- Discrete convolution (in particular you should be able to compute very simple,
  low-dimensional convolutions, like the toy example on slide 10)
- The basic premise of pooling (you won't be tested on subsampling and striding).
- CNN architectures (you should be able to describe a typical, example CNN
  architecture, like the one depicted on slide 22)
- Data augmentation (slide 24). We will not ask you questions about the
  distortion field method which starts on slide 25.
- Slides 25-38 are not in the exam material and can be ignored.
- From slide 40 you should know that it is possible to write a convolution
  operation as a matrix-vector multiplication, though we will not ask you to do
  it in the exam.
- Slides 41-end of slide deck 18 are not in the exam material. They are optional
  reading.

https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks

Some notes on pooling and subsampling:
- Pooling is an operation that produces a scalar from an mxm area of an input
  image. Common pooling operators are max pooling (we take the maximum of the
  values in the mxm area) and average pooling (we take the average of values in
  the area).
- The term subsampling is unfortunately overloaded in the literature and web
  discussions. In our slides (eg slide 19 of slide deck 18), subsampling refers
  to the use of non-overlapping pooling regions.
- However in the figure in slide 22 (which comes from Yann Lecun) the term
  "subsampling" is used to actually describe the strides used during pooling.
  In the case of Layer 2, 5x5 subsampling for a 10x10 pooling operation means
  that we consider overlapping pooling regions. In particular, we start by
  considering the top-left 10x10 region and perform our pooling operation (max
  or avg) to produce a scalar in the output. Then we shift the pooling region by
  5 pixels to the right (our horizontal stride) and perform the pooling
  operation again. If you do the math, you will see that for an input image of
  width 75 pixels, there are 14 different horizontal positions for our pooling
  region. Similarly there are 14 different vertical positions. Hence the
  dimension of the output is 14x14.
- Here is a relevant discussion on this striding for pooling layers:
  http://cs231n.github.io/convolutional-networks/#pool
- In the terminology of slides 18-20, "subsampling" refers to a stride that
  exactly matches the size of the pooling area (generally mxm) that's why they
  have this non-overlapping property.
- Other sources use the term "subsampling" for something significantly different
  (eg. a generalization of average pooling with learnable parameters:
  https://github.com/torch/nn/issues/944 ).
- As you see, terminology can be messy in a burgeoning discipline. Eventually,
  things settle down, but in the meantime we are faced with some naming
  conflicts and confusion. Clearly formulating and describing our operations
  (rather than using a name without definition) is the only sure way for precise
  technical communication.

RNN material that you should know (from slide deck 19):
- Basic discussion of language modeling in slide 3
- Structure of a basic RNN in slide 4
- How we unroll it to use it for an input text sequence (eg slide 5). Which
  parameters are shared across time in this diagram?
- Naming other applications on slide 9 (POS tagging, named entity recognition)
- How the loss is formed in RNNs (slides 12-13)
- How back propagation through time works (slide 14)
- What is the motivation of truncated BPTT and how does it work.
- Without having to know and explain the exact math, you should be able to
  describe the issue of exploding and vanishing gradient, and name the remedies
  of gradient clipping and orthogonal initialization.
- You should know that LSTMs are an important, powerful class of RNNs. Their
  main novelty is their use of memory cells. Memory cells carry a state, c, and
  have gates that control how much the new cell state is going to be influenced
  by its *input*, how well it is going to retain or *forget* the previously held
  state, and an *output* gate that dictates how much the RNN's hidden state h is
  going to be influence by the cell state. You should be able to describe this in
  words, but you do not have to remember the notation or exact connectivity of
  the LSTM cell.

https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks

Finally some general useful tips for NNs here:
https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-deep-learning-tips-and-tricks

