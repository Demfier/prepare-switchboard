# prepare-switchboard
This repository updates the originally written classes for [The Switchboard Dialog Act Corpus](https://compprag.christopherpotts.net/swda.html) and uses it to freely parse the dataset.

# Instructions to run
1. Extract the swda dataset inside `data/raw/`
2. Run `python main.py` to create train/val/test splits. The *_sentences.tsv files generated could be used to train an autoencoder while *_dialog.tsv files could used to train a simple sequence-to-sequence model for dialog generation.
