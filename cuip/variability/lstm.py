from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, LSTM, Embedding


def create_model(input_length):
    """DOCSTRING!"""
    # -- Create model.
    model = Sequential()
    model.add(Embedding(input_dim=188, output_dim=50, input_length=input_length))
    model.add(LSTM(output_dim=256, activation='sigmoid',
        inner_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(output_dim=256, activation='sigmoid',
        inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # -- Compile model.
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',
        metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # -- Load night
    lc.loadnight(lc.meta.index[5])
    # -- Format training data.
    y_train = np.array([ii == 1 for ii in lc.coords_cls]).astype(int)
    X_train = lc.lcs.T[np.array(lc.coords_cls.keys()) - 1]
    # -- Create lstm.
    model = create_model(len(X_train[0]))
    print ('Fitting model...')
    # -- Fit lstm.
    hist = model.fit(X_train, y_train, batch_size=64, nb_epoch=2,
        validation_split=0.1, verbose=1)
    # -- Load testing night.
    lc.loadnight(lc.meta.index[11])
    # -- Format testing data.
    y_test = np.array([ii == 1 for ii in lc.coords_cls]).astype(int)
    X_test = lc.lcs.T[np.array(lc.coords_cls.keys()) - 1]
    # -- Test lstm.
    score, acc = model.evaluate(X_test, y_test, batch_size=1)
    print('Test score:', score)
    print('Test accuracy:', acc)
