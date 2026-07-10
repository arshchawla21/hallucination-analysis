# LLM Entropy/Surprise for Realtime Hallucination Detection
> Exploring LLM hallucination from an algorithmic trading perspective.

As explored in previous work (HALT 2026), LLM **surprise** and **entropy** can act as a signal for  hallucination. LLMs are sampling machines at heart, predicting next token probabiltities. From this rises the notion of **surpise**,

$$-\log(P(x))$$

and **entropy**,

$$-\sum P(x_i)\log P(x_i)$$

In simple terms, these measure _"how unexpected a single specific token was"_ and _"the models total uncertainty"_.

## Idea
Prior work has explored the concept of identifying trends within these values in order to determine if an LLM is __hallucinating__. This is striking similar to the problem formulation of **algorthim trading**, and all time series analysis problems. 

In this work, we apply ideas from algorithmic trading in the hopes of identifing patterns in LLM uncertainty for hallucination detection.