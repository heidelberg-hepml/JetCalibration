import numpy as np
from iminuit import cost, Minuit
from sklearn import metrics


def get_max(values, weights):
    idx_max = np.argmax(weights)
    val_max = values[idx_max]

    std_max = np.sqrt(np.sum(weights*(values-val_max)**2)/np.sum(weights))
    
    fit_range = [val_max-std_max, val_max+std_max]
    mask_fit = (values >= fit_range[0]) & (values <= fit_range[1])

    ls = cost.LeastSquares(values[mask_fit], weights[mask_fit], weights[mask_fit]**0.5, gaussian)
    m = Minuit(ls, np.max(weights), val_max, std_max)
    m.migrad() # finds minimum of least_squares function
    m.hesse()  # accurately computes uncertainties

    return m.values[1], idx_max


def get_mean(values, weights):
    mean = np.average(values, weights=weights)
    idx_avg = np.argmin(np.abs(values-mean))
    return mean, idx_avg

def get_median(values, weights):
    
    # get index of median bin
    idx_srt = np.argsort(values)
    cum_sum = np.cumsum(weights[idx_srt])
    idx_med = idx_srt[np.searchsorted(cum_sum, 0.5*cum_sum[-1])]
    idx_low = idx_med-1 if idx_med > 1 else 0
    idx_hig = idx_med+1
 
    # interpolate between neighboring bins
    numer  = values[idx_med]*weights[idx_med]
    numer += values[idx_low]*weights[idx_low]
    numer += values[idx_hig]*weights[idx_hig]
    denom  = weights[idx_med]+weights[idx_low]+weights[idx_hig]
    median = np.divide(numer, denom, out=np.zeros_like(numer), where=(denom!=0))

    return median, idx_med

def get_asymm_errors(values, weights, idx_central):

    # 'idx_central': bin index of central value
    # calculate left/lower and right/upper errors
    err_low, wgt_low = 0.0, 0.0
    err_hig, wgt_hig = 0.0, 0.0

    for i in range(len(weights)):
        
        if i < idx_central: # sum up weights below the central value
            err_low += weights[i]*((values[i]-values[idx_central])**2)
            wgt_low += weights[i]
        
        elif i > idx_central: # sum up weights above the central value
            err_hig += weights[i]*((values[i]-values[idx_central])**2)
            wgt_hig += weights[i]
    
    if wgt_low == 0: err_low = 0 # calculate lower error
    else: err_low = np.sqrt(err_low/wgt_low)/np.sqrt(wgt_low)
        
    if wgt_hig == 0: err_hig = 0 # calculate upper error
    else: err_hig  = np.sqrt(err_hig/wgt_hig)/np.sqrt(wgt_hig)

    return err_low, err_hig

def get_iqr_range(values, weights, fsampl=0.68):
    # calculate IQR_68% from binned distribution:
    # width of the smallest of all ranges that contain 68 percent of all entries
    
    iqr_width = 0
    iqr_lower = 0
    iqr_upper = 0
        
    prob_prev = 1.0
    icont, ibin = 0, 0
    ibin_max, jbin_min = 0, 0
    dxprev = values[-1]-values[0]
    integral = np.sum(weights)

    if integral != 0:

        while (ibin < len(weights)) and (prob_prev >= fsampl):
            
            jbin = ibin
            icont = weights[ibin]
            while (jbin < len(weights)-1) and (icont/integral < fsampl):
                icont += weights[jbin]; jbin += 1
            iqr_lower = values[ibin]
            iqr_upper = values[jbin]
            iqr_width = iqr_upper-iqr_lower
            prob_prev = icont/integral

            if iqr_width < dxprev:
                dxprev, ibin_max, jbin_min = iqr_width, ibin, jbin
            ibin += 1
    
    iqr_lower = values[ibin_max]
    iqr_upper = values[jbin_min]
    iqr_width = iqr_upper-iqr_lower
    return iqr_width, iqr_lower, iqr_upper




def get_binned_statistics(hist, xedges, yedges, threshold=100):
    # important: len(binEdges) = len(binContents)+1 (nbins have nbins+1 edges)

    # NumPy histogram2d does not follow Cartesian convention,
    # therefore we transpose 'hist' for calculation purposes:
    hist = hist.T

    # mask for x-bin slices with enough statistics
    mask_sum = (np.sum(hist, axis=0) > threshold)
    xcenters = (xedges[:-1]+xedges[1:])/2
    ycenters = (yedges[:-1]+yedges[1:])/2

    statistics = {'bin_ctr': [], # x-bin centers
                  'bin_edg': [], # x-bin edges
                  'avg_val': [], # mean values
                  'avg_err': [[], []],
                  'med_val': [], # median values
                  'med_err': [[], []],
                  'max_val': [], # max values
                  'max_err': [[], []],
                  'iqr_rng': [], # min IQR range
                  'iqr_low': [],
                  'iqr_hig': []}
    
    mask_edges = np.insert(mask_sum, np.argmax(mask_sum==True), True)
    statistics['bin_ctr'] = xcenters[mask_sum]
    statistics['bin_edg'] = xedges[mask_edges]

    for c in range(hist.shape[1]): # loop through all x-bin slices
        if mask_sum[c]: # check if we enough statistics

            avg_val, avg_idx = get_mean(ycenters,   hist[:,c])
            med_val, med_idx = get_median(ycenters, hist[:,c])
            #max_val, max_idx = get_max(ycenters,    hist[:,c])

            avg_err_low, avg_err_hig = get_asymm_errors(ycenters, hist[:,c], avg_idx)
            med_err_low, med_err_hig = get_asymm_errors(ycenters, hist[:,c], med_idx)
            #max_err_low, max_err_hig = get_asymm_errors(ycenters, hist[:,c], max_idx)

            iqr_rng, iqr_low, iqr_hig = get_iqr_range(yedges, hist[:,c])
            
            statistics['avg_val'].append(avg_val)
            statistics['avg_err'][0].append(avg_err_low)
            statistics['avg_err'][1].append(avg_err_hig)
            statistics['med_val'].append(med_val)
            statistics['med_err'][0].append(med_err_low)
            statistics['med_err'][1].append(med_err_hig)
            #statistics['max_val'].append(max_val)
            #statistics['max_err'][0].append(max_err_low)
            #statistics['max_err'][1].append(max_err_hig)
            statistics['iqr_rng'].append(iqr_rng)
            statistics['iqr_low'].append(iqr_low)
            statistics['iqr_hig'].append(iqr_hig)

    return statistics


def standardize(x):
    '''
    Standardization rescales the data to have a mean of 0
    and a standard deviation of 1 (unit variance).
    '''
    mean, std = np.mean(x), np.std(x)
    out = (x-mean)/std
    return out, mean, std

def apply_save_log(x):
    epsilon = 1e-10
    minimum = np.min(x)
    if minimum <= 0:
        x = x-minimum+epsilon
    else:
        minimum = 0
        epsilon = 0
    return np.log10(x), minimum, epsilon

def compute_range(list, quantile = 0.0001):
    return [np.min([np.quantile(arr, quantile) for arr in list]),  np.max([np.quantile(arr, 1-quantile) for arr in list])]

def get_ratio(density1, density2):
    density1_nonzero = density1[(density1 != 0.0) & (density2 != 0.0)]
    density2_nonzero = density2[(density1 != 0.0) & (density2 != 0.0)]

    ratio = np.full_like(density1, np.nan)
    ratio[(density1 != 0.0) & (density2 != 0.0)] = density1_nonzero / density2_nonzero
    ratio[(density1 == 0.0) & (density2 != 0.0)] = 0.0
    ratio[(density1 != 0.0) & (density2 == 0.0)] = np.inf
    ratio[(density1 == 0.0) & (density2 == 0.0)] = 0.0 #1.0
    
    return ratio

def gaussian(x, norm, mu, sigma):
    return norm*np.exp(-(x-mu)**2/(2*sigma**2))

def split_gaussian(xs, norm, mu, sig1, sig2):
    f = []
    for x in xs:
        if x < mu: f.append(gaussian(x, norm, mu, sig1))
        else: f.append(gaussian(x, norm, mu, sig2))
    return np.array(f)

def find_nearest(array, value):
    array = np.asarray(array)
    idx   = np.argmin(np.abs(array-value))
    return array[idx]

def get_perf_stats(lbs, scr):
    acc = metrics.accuracy_score(lbs, (scr>=0.5).astype(np.float32))
    auc = metrics.roc_auc_score(lbs, scr)
    fpr, tpr, thresholds = metrics.roc_curve(lbs, scr)
    fpr2 = [fpr[i] for i in range(len(fpr)) if tpr[i]>=0.5]
    tpr2 = [tpr[i] for i in range(len(tpr)) if tpr[i]>=0.5]
    try:
        imtafe = np.nan_to_num(1/fpr2[list(tpr2).index(find_nearest(list(tpr2), 0.5))])
    except:
        imtafe = 1
    return acc, auc, imtafe

def first_nonzero(arr, axis=0, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(arr, axis=0, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)