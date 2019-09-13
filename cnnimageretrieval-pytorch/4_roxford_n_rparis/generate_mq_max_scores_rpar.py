import numpy as np
import pickle
import math


# load score file
# roxford
#f = 'roxf_single.npy'
# rparis
f = 'rpar_single.npy'

# mq-max
# roxford
#nf = 'roxf_mq_max.npy'
# paris
nf = 'rpar_mq_max.npy'

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
n_queries = scores.shape[1]
print (n_queries)

gnd_fname = '/app/cnnimageretrieval-pytorch/data/test/rparis6k/gnd_rparis6k.pkl' 
with open(gnd_fname, 'rb') as f:
  cfg = pickle.load(f)

# build query groups
# key: group id, value: list => 'easy' gt data
q_group = {}

q_group[0] = cfg['gnd'][0]['easy']

group_id = 1
for i in range (1, n_queries):
  if cfg['gnd'][i]['easy'] in q_group.values():
    print ('this is already in the query group.')
  else:
    q_group[group_id] = cfg['gnd'][i]['easy']
    group_id += 1

#print (q_group)

# assigned group
g_members = {k: [] for k in range(len(q_group))}

# assign each query to the query group
for i in range (n_queries):
  # group 0
  if cfg['gnd'][i]['easy'] == q_group[0]:
    g_members[0].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[1]:
    g_members[1].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[2]:
    g_members[2].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[3]:
    g_members[3].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[4]:
    g_members[4].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[5]:
    g_members[5].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[6]:
    g_members[6].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[7]:
    g_members[7].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[8]:
    g_members[8].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[9]:
    g_members[9].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[10]:
    g_members[10].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[11]:
    g_members[11].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[12]:
    g_members[12].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[13]:
    g_members[13].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[14]:
    g_members[14].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[15]:
    g_members[15].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[16]:
    g_members[16].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[17]:
    g_members[17].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[18]:
    g_members[18].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[19]:
    g_members[19].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[20]:
    g_members[20].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[21]:
    g_members[21].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[22]:
    g_members[22].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[23]:
    g_members[23].append(i)
  elif cfg['gnd'][i]['easy'] == q_group[24]:
    g_members[24].append(i)

#print (g_members)


# compute mq-avg per each query group
for g_id, q_list in g_members.items():

  # key: image, value: accumulated_score
  score_dict = {}

  # for each query group
  for m_idx in q_list:
    #print (m_idx)
    for img in range(n_imgs):
      if img in score_dict:
        if (scores_t[m_idx][img] > score_dict[img]):
          score_dict[img] = scores_t[m_idx][img]
      else: 
        score_dict[img] = scores_t[m_idx][img]

  for img_idx, a_score in score_dict.items():
    for m_idx in q_list:
      new_scores_t[m_idx][img_idx] = a_score


new_scores = new_scores_t.T
np.save(nf, new_scores)
print ('done!')

