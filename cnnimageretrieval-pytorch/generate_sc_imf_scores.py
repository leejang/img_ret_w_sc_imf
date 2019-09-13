import numpy as np
import math

# load score file
# oxford
#f = 'oxf_single.npy'
# paris
f = 'par_single.npy'

# sc-imf
# oxford
#nf = 'oxf_sc_imf.npy'
# paris
nf = 'par_sc_imf.npy'

# load
scores = np.load(f)
print (scores.shape)

scores_t = scores.T
print (scores_t.shape)

# new scores
new_scores_t = np.empty_like(scores_t)
print (new_scores_t.shape)

n_imgs = scores.shape[0]
print (n_imgs)

# for Sample Confidence with Inverse Model Frequency (IMF)
#N = n_imgs

#################
# oxford
# 95.44
#N = 330
#mf_thr = 0.09

# 95.95
#N = 308
#mf_thr = 0.11

# 96.19
#N = 302
#mf_thr = 0.12

# 96.08
#N = 290
#mf_thr = 0.13

#################
# paris
# 94.55 (0.1)
# 94.61 (0.09)
# 94.69 (0.08)
# 94.73 (0.07)
# 94.75 (0.05)
# 94.76 (0.06)

# 95.29
#N = 968
#mf_thr = 0.055

# 95.25
#N = 871
#mf_thr = 0.06

# 95.33
#N = 1046
#mf_thr = 0.052

N = 1077
mf_thr = 0.051

mf = 0

# key: query, value: weight: weight dict
w_dict = {}

for idx, score in np.ndenumerate(scores_t):

  # compute model frequency (mf)
  if (score > mf_thr):
    mf += 1

  if (idx[1] % (n_imgs) == (n_imgs - 1)):
    print (mf)
    if (mf == 0):
      w_dict[idx[0]] = math.log10(N/float(1))
    else:
      w_dict[idx[0]] = math.log10(N/float(mf))
    mf = 0

print (w_dict)


# key: image, value: accumulated_score
score_dict = {}

m_idx = 0

for idx, score in np.ndenumerate(scores_t):

  if (idx[0] % 5 == 4) and (idx[1] % (n_imgs) == (n_imgs - 1)):
    # the last image given the last query for the same building
    #score_dict[idx[1]] += (score - mf_thr) * w_dict[idx[0]]
    if ((score - mf_thr) * w_dict[idx[0]] > score_dict[idx[1]]): 
      score_dict[idx[1]] = (score - mf_thr) * w_dict[idx[0]]

    for img_idx, a_score in score_dict.items():
      # compute average score
      #a_score /= 5
      new_scores_t[m_idx * 5 + 0][img_idx] = a_score
      new_scores_t[m_idx * 5 + 1][img_idx] = a_score
      new_scores_t[m_idx * 5 + 2][img_idx] = a_score
      new_scores_t[m_idx * 5 + 3][img_idx] = a_score
      new_scores_t[m_idx * 5 + 4][img_idx] = a_score

    m_idx += 1
    print (idx, score)
    print (m_idx)
    score_dict.clear()

  else:
    if idx[1] in score_dict:
      #score_dict[idx[1]] += (score - mf_thr) * w_dict[idx[0]]
      if ((score - mf_thr) * w_dict[idx[0]] > score_dict[idx[1]]): 
        score_dict[idx[1]] = (score - mf_thr) * w_dict[idx[0]]
    else:
      score_dict[idx[1]] = (score - mf_thr) * w_dict[idx[0]]


new_scores = new_scores_t.T
np.save(nf, new_scores)

print ('done!')

