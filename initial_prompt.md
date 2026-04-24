I want you to train a super tiny GPT model from scratch, that I want to use for demonstration purposes. It should be trained on ordinary english text, but it should be extremely small, to make it easy to see how the attention and MLP layers etc. work together to generate output.

We want to train a single decode layer GPT model that is trained on an adequate corpus. It should be predicting English text, though we could make some simplifications, such as all-lower-case, or reduced set of special characters (e.g. only commas and periods; or only periods). It should have a single attention layer (though with possibly multiple heads) so we can visualize clearly which previous tokens are incorporated into the next token prediction. Maybe even have something like sparse attention, where only the top 3 or top 5 or so previous values, per head, are incorporated, the others are zeroed: This would make interpretation even more straightforward: one could see that, in a given position, precisely which other tokens were incorporated.

Apart from this, we can go all out: use the best activation function (try a few), use innovations like muon or mHC, try out various things that may work well. See some sources on NanoGPT speedruns or results from Karpathy's AutoResearch what might be good ideas.

I want you to do a systematic study and investigation into what the best kind of setup for this is: What kind of tokenization (letter, word, bpe, others?), what kind of simplification of the language helps (e.g. all lower case etc?; having at least commas and periods would be nice, but if removing them makes output that much better it is also ok.), what kind of other tricks, as long a it is a single attention layer with various heads that is noticeably a GPT / next token predictor for something recognizably english language. Use an adequate corpus; try to not go beyond 200 gb of disk space usage (probably should be much smaller though? Quality probably matters a lot here). We care about output quality for a given small architecture, so the model will likely profit from being 'over-trained' relative to compute-optimal scaling.

You are running in a user-mode podman container with 16 gb of ram on an AMD laptop; see if you can access some compute accelerator, but just use the 16 cpu cores otherwise. You can do anything in here that the sandbox allows.

Your plan is to:
1. investigate the environment you are running in, make decisions w/r/t deep learning framework, corpus, evaluation 
2. investigate what possible architectural decisions you should try out, e.g. what tricks should be tried to make the small model still perform relatively well (make only very small experiments at this point)
3. iteratively and systematically make experiments to narrow down on an architecture that makes sense. You can spend hours on this. You should probably make small runs at first, and should likely try some HPO framework with early stopping and increasing budget / multifidelity functionality for this
4. Train the final model when you are sure there is no more performance that can be gained, when you know we are at the edge of what is possible given the constraints.

Keep a notepad / research log in RESEARCH.md where you document your state of knowledge, open and answered questions, further ideas that you have tried or are planning to try etc. This document should be progressively updated as your knowledge evolves and the research progresses.

Work on this for hours and hours. Do not stop. Always continue and try to find something better until you are absolutely sure there is no more benefit to be had.
