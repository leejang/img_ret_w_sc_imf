import numpy as np

# load score file
# oxford
#f = 'oxf_single.npy'
# paris
#f = 'par_single.npy'

# roxford
f = 'roxf_single.npy'
# rparis
#f = 'rpar_single.npy'

# mq-avg
# oxford
#nf = 'oxf_mq_avg.npy'
# paris
#nf = 'par_mq_avg.npy'

# roxford
nf = 'roxf_mq_avg.npy'
# paris
#nf = 'rpar_mq_avg.npy'

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

# key: image, value: accumulated_score
score_dict = {}

m_idx = 0

for idx, score in np.ndenumerate(scores_t):

  if (idx[0] % 5 == 4) and (idx[1] % (n_imgs) == (n_imgs - 1)):
    # the last image given the last query for the same building
    score_dict[idx[1]] += score

    for img_idx, a_score in score_dict.items():
      # compute average score
      a_score /= 5
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
      score_dict[idx[1]] += score
    else:
      score_dict[idx[1]] = score


new_scores = new_scores_t.T
np.save(nf, new_scores)

print ('done!')

