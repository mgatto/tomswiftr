# Finding "Tom Swifties": Part of Speech Puns in the Tom Swift Sci-Fi Novellas of the 1910s
Github: https://github.com/mgatto/TomSwifter

My proposal is to enhance the results of this paper:

T. Litovkina, A. (2014). “I see,” said Tom icily: Tom Swifties at the beginning of the 21st century. *The European Journal of Humour Research*, 2(2), 54-67. https://doi.org/10.7592/EJHR2014.2.2.tlitovkina

"Tom Swifties" are a pun construction where an adverb modifies a verb denoting a speech act performed by the main character, Tom Swift. The pun works because, while the adverb is dependent on its immediately preceding verb, its semantic meaning may also apply to a preceding quote or description of the main character's action within the same sentence. For example,

“Is your name Frank Lee?” Tom asked frankly

The adverb frankly invokes the preceding word Frank within the same sentence. The pun mechanics may differ and not always rely on spelling; it may also rely on homophones which I think may be more difficult to detect in a term project.

Nevertheless, one extension to this project may be to classify the puns and see if their proportions change over the course of the novellas' publication dates. This becomes more relevant as we learn that the author's name was a pen name and represented several individuals over the serial's lifetime. This will only be possible if the project is completed before the due date :-)

## Data Source

To enhance Litovkina's results, this project will parse the raw texts of the first 10 Tom Swift novellas, published between 1910 and 1911. These texts are legally and publicly available at Project Gutenberg.

## Methodology

Python code using SpaCy tools will parse the text and automatically find occurrences of Tom Swifties in the text. They will be counted and categorized according to Litovkina's classifications. I will divide the books into sets of 70%-20%-10%: 7 of the 10 novellas' text will be training data, 2 of 10 will be test data and 1 will be development data. The data will be serialized into an appropriate disk-storable format.

## Judging Results: Recall and Accuracy

I will manually scan the pos-tagged data of the two test texts to find all occurrences of a speech verb such as "said", etc combined with an adverb. I will tag each such occurrence and judge myself if it is a pun. A larger study would use several human classifiers to product this Gold Standard.

I do not currently believe that the century time-span between the novella's writings and SpaCy's large English model will impact the results.