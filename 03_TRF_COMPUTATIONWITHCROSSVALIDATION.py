# specify parameters 
days = ["day1", "day2"]
band = "deltatheta"

# load modules 
import mne
from scipy import stats
import numpy as np
np.set_printoptions(threshold=10000)
import pickle
import mat73
import os
from sklearn.model_selection import KFold

for day in days: 
    
    # path to data files
    data_path = "D:/data_mne/"
    
    # subject
    subj = ['Bou_Ni']
    sub_idx = 0 
    
    # list of conditions
    if day == "day1":
        condition_list = ['produce_music', 'perceive_music_produced', 'perceive_music_new', 'perceive_music_newrepetition', 'produce_speech', 'perceive_speech_produced', 'perceive_speech_new', 'perceive_speech_newrepetition']
        pairs = [["music","produce_music"], ["music", "perceive_music_produced"], ["music_new", "perceive_music_new"], ["music_new", "perceive_music_newrepetition"],
                 ["speech", "produce_speech"], ["speech", "perceive_speech_produced"], ["speech_new", "perceive_speech_new"], ["speech_new", "perceive_speech_newrepetition"]]
    
    if day == "day2":
        condition_list = ['produce_music', 'perceive_music_produced', 'produce_speech', 'perceive_speech_produced']
        pairs = [["music","produce_music"], ["music", "perceive_music_produced"], ["speech", "produce_speech"], ["speech", "perceive_speech_produced"]]
    
    
    # matlabfiles path
    features_path = data_path + subj[sub_idx] + "/stimulusfeatures/"
    
    # dictionary path
    dictionary_path = data_path + subj[sub_idx] + "/dictionary"
    
    #### load in epochs
    epochs = {}
    
    for condition in condition_list:
        print(condition)
        preprocessed_path = data_path + subj[sub_idx] + "/preprocessed/" + condition + "/"
        print(preprocessed_path)
    
        for files in os.listdir(preprocessed_path):
            filename = day + "_bipolar_epochs_preprocessed_test.fif"
            if filename in files:
                path = preprocessed_path + files + '/'
                epochs[condition] = mne.read_epochs(path, preload=True)
                print(epochs[condition])
                if band == "deltatheta":
                    l_freq = 1
                    h_freq = 8    
                if band == "beta":
                    l_freq = 13
                    h_freq = 30
                if band == "lowgamma":
                    l_freq = 30 
                    h_freq = 80 
                if band == "highgamma": 
                    l_freq = 80
                    h_freq = 180
    
                epochs[condition].filter(l_freq=l_freq, h_freq=h_freq, n_jobs=-1)  # low-pass filter
    
    #### import matlab files with stimulus features
    stimulus_features = {}
    
    for condition in condition_list:
        if (condition == 'produce_music') or (condition == 'perceive_music_produced'):
            if day == "day1":
                matlab_file = features_path + 'seq_BouNi_musique-001-20220211144930_auddims_clean.mat'
            if day == "day2":
                matlab_file = features_path + 'AuditoryDimensions_48kHz_play_clean.mat'
                print(matlab_file)
            features = mat73.loadmat(matlab_file, 'r')
            stimulus_features['music'] = features
    
        elif (condition == 'perceive_music_new') or (condition == 'perceive_music_newrepetition'):
            matlab_file = features_path + 'Telemann_Fantasias_7m15s_auddims_clean.mat'
            features = mat73.loadmat(matlab_file, 'r')
            stimulus_features['music_new'] = features
    
        elif (condition == 'produce_speech') or (condition == 'perceive_speech_produced'):
            if day == "day1":
                matlab_file = features_path + 'seq_BouNi_Reading-001-20220211150422_auddims_clean.mat'
            if day == "day2":
                matlab_file = features_path + 'AuditoryDimensions_48kHz_read_clean.mat'
                print(matlab_file)
    
            features = mat73.loadmat(matlab_file, 'r')
            stimulus_features['speech'] = features
    
        elif (condition == 'perceive_speech_new') or (condition == 'perceive_speech_newrepetition'):
            matlab_file = features_path + 'FideleJean_7m_auddims_clean.mat'
            features = mat73.loadmat(matlab_file, 'r')
            stimulus_features['speech_new'] = features
    
    if day == "day1":
        stimulus_features_list = ['music', 'music_new', 'speech', 'speech_new']
    
    if day == "day2":
        stimulus_features_list = ['music', 'speech']
    
    # define parameters
    fs = 100
    tmin = epochs['perceive_speech_produced'].tmin
    tmax = epochs['perceive_speech_produced'].tmax
    n_timepoints = 29750
    n_trials = 1
    
    regressors = {}
    
    #### define dimensions of regressors -> (n_timepoints, n_trials, n_regressors)
    for stimulus in stimulus_features_list:
        regressors[stimulus] = {}
    
        labels = stimulus_features[stimulus]['names']
        X = stimulus_features[stimulus]['X']
    
        # get dimensions
        X = np.tile(X[:, :, np.newaxis], (1, n_trials, 1))
        X = np.moveaxis(X, [0, 1, 2], [0, 2, 1])  
    
        # downsample regressors
        X = mne.filter.resample(X, down=stimulus_features[stimulus]['fs'] / fs, npad='auto', axis=0)
    
        # crop to desired number of time points
        X = X[:n_timepoints, :, :]
    
        # correct time axis if necessary
        if tmin < 0:
            X = X[0: int((tmax + tmin) * fs), :, :]
        else:
            X = X[0: int((tmax) * fs), :, :]
    
        # calculate derivatives
        labels_dif = []
        for regr in range(0, 5):
            dif = np.diff(X[:, :, regr], axis=0)
            dif = np.concatenate((np.zeros((1, n_trials)), dif), axis=0)
            if regr == 0:
                dif = np.where(dif < 0, 0, dif)
            dif = np.expand_dims(dif, axis=2)
    
            label_dif = labels[regr] + "_derivative"
            labels_dif.append(label_dif)
    
            # add derivative to regressors
            X = np.concatenate((X, dif), axis=2)
    
        # add new labels
        labels += labels_dif
    
        # z-score regressors
        X = stats.zscore(X, axis=0)
    
        r_select = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # select regressors
        X, labels = X[:, :, r_select], [labels[i] for i in r_select]
    
        n_timepoints_new = np.shape(X)[0]
    
        regressors[stimulus] = (X, labels)
    
        del X
    
    # define picks
    all_picks = epochs['produce_music'].ch_names
    
    picks = []
    channel_index = np.r_[42:50, 89:96]
    for index in channel_index: 
        picks.append(all_picks[index])

    fs = 100
    seeg = {}
    
    #### define dimensions of seeg data, dimensions should be : (n_timepoints, n_trials, n_channels)
    for condition in condition_list:
        seeg[condition] = {}
    
        Y = epochs[condition].get_data(
            picks=picks)  # dimensions when loaded : n_trials, n_channels, n_times # (picks=picks)
    
        # get right dimensions
        Y = np.transpose(Y, (2, 0, 1))  # transpose to : n_times, n_trials, n_channels
    
        # downsample eeg data
        Y = mne.filter.resample(Y, down=epochs[condition].info['sfreq'] / fs, npad='auto', axis=0)
    
        # crop to desired number of time points
        Y = Y[:n_timepoints, :, :]
    
        # correct time axis if necessary
        if epochs[condition].tmin < 0:  # make sure that epochs start at t = 0
            Y = Y[int(-epochs[condition].tmin * fs): int(epochs[condition].tmax * fs), :, :]
    
        # z-score seeg data
        Y = stats.zscore(Y, axis=0)
    
        seeg[condition] = Y
    
        del Y
    
    # store seeg and regressors
    with open(dictionary_path + f"/TRF_regressors_{band}_{day}.pickle", 'wb') as f:
        pickle.dump(regressors, f)
    
    with open(dictionary_path + f"/TRF_seeg_{band}_{day}.pickle", 'wb') as f:
        pickle.dump(seeg, f)
    
    
    ########## TRF ############
    scoring = 'r2'
    #alphas = np.logspace(2, 6, 5)
    alphas = [1e-6, 1e-4, 1e-2, 0, 1e+2, 1e+4, 1e+6]
    tw = [-0.2, 0.5]                                            
    times = np.linspace(tw[0],tw[1], int(np.diff(tw)*fs +1))    
    
    TRF = {}
    
    n_folds_outer = 20
    n_folds_inner = 2
    n_chunk = 20
    
    #### for-loop: condition and regressor pair #####
    for pair in pairs:
        condition = pair[1]
        TRF[condition] = {}
    
        #### for-loop: channel #####
        for i_chan in range(len(picks)):
            TRF[condition][picks[i_chan]] = {}
    
            # data
            X = regressors[pair[0]][0]
            Y = seeg[pair[1]]
    
            # chunking data into trials
            chunk_size = np.floor(X.shape[0] / n_chunk).astype(int)
            tmp_X = np.full((chunk_size, n_chunk, X.shape[2]), None)
            tmp_Y = np.full((chunk_size, n_chunk, Y.shape[2]), None)
            for i_fold in range(n_chunk):
                tmp_X[:, i_fold, :] = X[i_fold * chunk_size:(i_fold + 1) * chunk_size, 0, :]
                tmp_Y[:, i_fold, :] = Y[i_fold * chunk_size:(i_fold + 1) * chunk_size, 0, :]
            X, Y = tmp_X.astype(float), tmp_Y.astype(float)
    
            # objects to store results of outer loop
            best_alphas = []
            scores_cv = np.zeros((n_folds_outer))
            coefs_cv = np.zeros((n_folds_outer, X.shape[2], times.shape[0]))
            scores_fit = np.zeros((n_folds_outer))
            coefs_fit = np.zeros((n_folds_outer, X.shape[2], times.shape[0]))
    
            # outer cross-validation
            outer_cv = KFold(n_splits=n_folds_outer, shuffle=True, random_state=37)
            for i_fold_outer, (outer_train, outer_test) in enumerate(outer_cv.split(np.moveaxis(X, 1, 0))):
    
                # data for inner cross-validation
                X_train, X_test = X[:, outer_train, :], X[:, outer_test, :]  # split data in test and training data
                Y_train, Y_test = Y[:, outer_train, :], Y[:, outer_test, :]
        
                # variables to store the results for inner loop
                scores_val_all = np.zeros((len(alphas), n_folds_inner))
                models = [[] for _ in range(len(alphas))]
                Y_chan = np.expand_dims(Y_train[:, :, i_chan], 2)
                Y_chan_test = np.expand_dims(Y_test[:, :, i_chan], 2)
                predicted_Y_fit = np.zeros((n_folds_outer, Y_test.shape[0]))

                #### for-loop: alpha #####
                for ii, alpha in enumerate(alphas):
                    in_score = np.zeros(n_folds_inner)
    
                    # inner cross-validation
                    inner_cv = KFold(n_folds_inner, shuffle=True, random_state=37)
                    for i_fold_inner, (train, test) in enumerate(inner_cv.split(np.moveaxis(X_train, 1, 0))):
    
                         # define model
                        rf = mne.decoding.ReceptiveField(tw[0], tw[1], fs, feature_names=labels, estimator=alpha,
                                                            scoring=scoring, n_jobs=-1)
    
                         # fit on train data
                        rf.fit(X_train[:, train, :], Y_chan[:, train, :])

                        # store score & model at each cv
                        in_score[i_fold_inner] = np.mean(rf.score(X_train[:, test, :], Y_chan[:, test, :]))
                        models[ii].append(rf)
    
                    # get scores
                    scores_val_all[ii] = in_score
    
                # choose model that performed best on test data
                ix_best_alpha = np.argmax(np.mean(scores_val_all, axis=1))
                best_alpha = alphas[ix_best_alpha]  # retrieve best alpha for this channel
    
                # store averaged coefs and scores for best alpha
                coefs_cv[i_fold_outer] = np.array([models[ix_best_alpha][i_fold].coef_ for i_fold in range(n_folds_inner)]).mean(axis=0)  # mean of coefs with best alpha across inner folds
                scores_cv[i_fold_outer] = np.mean(scores_val_all[ix_best_alpha])
    
                # fit another model entire outer loop train data and test on outer loop test data
                rf = mne.decoding.ReceptiveField(tw[0], tw[1], fs, feature_names=labels, estimator=best_alpha,
                                                     scoring='r2', n_jobs=-1)
                rf.fit(X_train, Y_chan)
                coefs_fit[i_fold_outer] = rf.coef_
                scores_fit[i_fold_outer] = rf.score(X_test, Y_chan_test)
                predicted_Y_fit[i_fold_outer] = np.squeeze(rf.predict(X_test))
                times = rf.delays_ / fs
    
            TRF[condition][picks[i_chan]] = {
                'score': scores_fit,
                'coefs': coefs_fit,
                'predicted_Y': predicted_Y_fit,
                'times': times
            }
    
            # store data
            with open(dictionary_path + '/' + str(condition) + "_" + str(
                    picks[i_chan]) + "_" + band + "_" + day + "_test_TRF_results_crossvalidate.pickle", 'wb') as f:
                pickle.dump(TRF, f)