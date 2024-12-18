In this project I train a language model specialized in
performing punctuation in the spanish language. 

The motivation is that usually dictation tools can't capitalize nor
punctuate appropriately. I hope to produce a model that can consume the output
of a dictation model and correct it  by adding punctuation.

The first attempt will be by using paragraphs of the spanish wikipedia
and use transfer learning with the GPT2-spanish model, and train it in a taks of sequence-to-sequence.
