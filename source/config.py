EMBEDDING_DIMENSION = 100
#EMBEDDING_FILE_LOC = 'C:/Users/jordanj/Desktop/ToxicComentAnalysisProject/model/glove/glove.6B.' + str(EMBEDDING_DIMENSION) + 'd.txt'
TRAINING_DATA_LOC = 'C:/Users/jordanj/Desktop/ToxicComentAnalysisProject/dataset/train.csv'
TEST_DATA_LABEL = 'C:/Users/jordanj/Desktop/ToxicComentAnalysisProject/dataset/test_labels.csv'
TEST_DATA_COMMENTS = 'C:/Users/jordanj/Desktop/ToxicComentAnalysisProject/dataset/test.csv'
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100
BATCH_SIZE = 128
EPOCHS = 30
VALIDATION_SPLIT = 0.2
DETECTION_CLASSES = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate',
    'neutral']
MODEL_LOC = 'C:/Users/jordanj/Desktop/ToxicComentAnalysisProject/model/comments_toxicity.h5'
TOKENIZER_LOC = 'C:/Users/jordanj/Desktop/ToxicComentAnalysisProject/model/tokenizer.pickle'