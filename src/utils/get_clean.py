
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import pickle


def get_output(output ,mode) : 
    out = output[mode]['outputs']
    output = [[key, v['original_labels'].item(), v['noisy_labels'], v['loss']] for key, v in out.items()]
    output_df = pd.DataFrame(output, columns=['key', 'original_labels', 'noisy_labels', 'loss'])
    output_df['is_clean'] = (output_df['original_labels'] == output_df['noisy_labels']).astype(int)
    return output_df

def fit_gmm_candidates(X, k_min=1, k_max=5, n_init=10):
    ks, bics, aics, models = [], [], [], []
    for k in range(k_min, k_max+1):
        gm = GaussianMixture(
            n_components=k, covariance_type='full',
            n_init=n_init, reg_covar=1e-6, random_state=0
        ).fit(X.reshape(-1, 1))
        ks.append(k)
        bics.append(gm.bic(X.reshape(-1,1)))
        aics.append(gm.aic(X.reshape(-1,1)))
        models.append(gm)
    return np.array(ks), np.array(bics), np.array(aics), models

def _smooth(y, w=3):
    if w <= 1: return y.copy()
    y_pad = np.r_[y[0], y, y[-1]]
    return np.convolve(y_pad, np.ones(w)/w, mode='same')[1:-1]

def elbow_by_segmented(k, y, min_len=2):
    # try all split points s with at least min_len on both sides
    best_s, best_sse = None, np.inf
    x = k.astype(float)
    for s in range(min_len, len(k)-min_len):
        # left fit
        xl, yl = x[:s], y[:s]
        xr, yr = x[s:], y[s:]
        Al = np.c_[xl, np.ones_like(xl)]
        Ar = np.c_[xr, np.ones_like(xr)]
        bl, *_ = np.linalg.lstsq(Al, yl, rcond=None)
        br, *_ = np.linalg.lstsq(Ar, yr, rcond=None)
        sse = np.sum((yl - Al@bl)**2) + np.sum((yr - Ar@br)**2)
        if sse < best_sse:
            best_sse, best_s = sse, s
    return k[best_s], best_s, best_sse


def detect_elbow(k, aic_or_bic):
    y = np.asarray(aic_or_bic, dtype=float)
    return elbow_by_segmented(np.asarray(k), y, min_len=2)

def df_process(output) : 
    train_df = get_output(output, 'train')
    label0_loss_values = train_df.query('noisy_labels == 0')['loss'].values.reshape(-1, 1)
    label1_loss_values = train_df.query('noisy_labels == 1')['loss'].values.reshape(-1, 1)
    
    ks, bics, _, _ = fit_gmm_candidates(label0_loss_values, k_min=1, k_max=5, n_init=10)
    label0_k_elbow, _, _ = detect_elbow(ks, bics)
    print(f"Label 0 GMM K Elbow : {label0_k_elbow}")
    
    ks, bics, _, _ = fit_gmm_candidates(label1_loss_values, k_min=1, k_max=5, n_init=10)
    label1_k_elbow, _, _ = detect_elbow(ks, bics)
    print(f"Label 1 GMM K Elbow : {label1_k_elbow}")
        
    gmm1 = GaussianMixture(n_components=label0_k_elbow, random_state=42)
    gmm1.fit(label0_loss_values)
    gmm1_comp = gmm1.predict(label0_loss_values)
    # clean component is the one with the lowest mean loss
    clean_component = np.argmin(gmm1.means_)
    group1_cutoff = np.mean(label0_loss_values[gmm1_comp == clean_component])

    gmm2 = GaussianMixture(n_components=label1_k_elbow, random_state=42)
    gmm2.fit(label1_loss_values)
    gmm2_comp = gmm2.predict(label1_loss_values)
    # clean component is the one with the lowest mean loss
    clean_component = np.argmin(gmm2.means_)
    group2_cutoff = np.mean(label1_loss_values[gmm2_comp == clean_component])

    train_df['predicted_clean'] = train_df.apply(lambda x : x['loss'] <= group1_cutoff if x['noisy_labels'] == 0 else x['loss'] <= group2_cutoff, axis=1).astype(int)
    
    valid_df = get_output(output, 'valid')
    valid_df['predicted_clean'] = valid_df.apply(lambda x : x['loss'] <= group1_cutoff if x['noisy_labels'] == 0 else x['loss'] <= group2_cutoff, axis=1).astype(int)
    
    test_df = get_output(output, 'test')
    test_df['predicted_clean'] = test_df.apply(lambda x : x['loss'] <= group1_cutoff if x['noisy_labels'] == 0 else x['loss'] <= group2_cutoff, axis=1).astype(int)
    return train_df, valid_df, test_df