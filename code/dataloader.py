import torch, random
from torchtext import data, datasets

def get_data(batch_size,glove=False, root='data/.data'):

    if glove:
        TEXT = data.Field(tokenize='spacy')
    else:
        TEXT = data.Field()
        

    LABEL = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root)

    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of testing examples: {len(test_data)}')

    train_data, valid_data = train_data.split()

    print(f'\nNumber of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')

    if glove:
        TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
    else:
        # TODO: 
        TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")

    LABEL.build_vocab(train_data)

    print(f"\nUnique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

    print(f"\nNumber of Input Dimensions: {len(TEXT.vocab)}")

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size=batch_size,
        device=device)
    return train_iterator, valid_iterator, test_iterator, TEXT.vocab.vectors



if __name__ == '__main__':
    get_data(128)