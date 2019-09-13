k = 200;

% load gnd file
% oxford
gnd_file = 'data/gnd_oxford5k.mat';
load(gnd_file);

% oxford5k scores
load('scores_oxf', 'scores')

% paris
%gnd_file = 'data/gnd_paris6k.mat';
%load(gnd_file);

% paris6k scores
%load('scores_par', 'scores')

% cell array to store new scores after applying sc-imf
new_scores = scores;

% Hash table (key: query, value: weight)
M = containers.Map;

% oxford5k
% 96.66
%N = 1362;
%mf_thr = 300;

% 96.65
%N = 1757;
%mf_thr = 250;

% 96.67
%N = 1047;
%mf_thr = 350;

% 96.67
%N = 814;
%mf_thr = 450;

% 96.67
N = 500;
mf_thr = 500;

% 96.66
%N = 4159;
%mf_thr = 100;

% paris6k
% 97.12
%N = 498;
%mf_thr = 500;
%N = 465;
%mf_thr = 600;
%N = 391;
%mf_thr = 700;

% 97.13
%N = 395;
%mf_thr = 800;
%N = 800;
%mf_thr = 200;

% 97.10
%N = 328;
%mf_thr = 1100;

% 97.16
%N = 2240;
%mf_thr = 100;

union_set = [];

% to get a new N with union set
% Query
for q = 1:numel(gnd); % number of queries

  % image matching scores given each query
  img_scr = scores(1,q);
  [t, n_img] = size(img_scr{1});

  mf = 0;
  cur_set = [];

  for img = 1:n_img
    scr = img_scr{1}(img);
    if (scr > mf_thr)
      mf = mf + 1;
      cur_set = [cur_set, img];
      union_set = union(union_set, cur_set); 
    end
  end
end

N = length(union_set);
disp(N);

% Query
for q = 1:numel(gnd); % number of queries

  % image matching scores given each query
  img_scr = scores(1,q);
  [t, n_img] = size(img_scr{1});

  mf = 0;

  for img = 1:n_img
    scr = img_scr{1}(img);
    if (scr > mf_thr)
      mf = mf + 1;
      cur_set = [cur_set, img];
      union_set = union(union_set, cur_set); 
    end
  end

  fprintf('q %d, mf: %d\n', q, mf);

  if (mf == 0)
    M(num2str(q)) = log10(N/(1));
  else
    M(num2str(q)) = log10(N/(mf));
  end

end


% check the keys and values of the hash table
%disp(keys(M));
%disp(values(M));

% Hash table for scores (key: img, value: weighted scores with sc-imf)
% Score Map
S_M = containers.Map;

m_idx = 0;

% Query
for q = 1:numel(gnd); % number of queries

  % image matching scores given each query
  img_scr = scores(q);
  [t, n_img] = size(img_scr{1});

  for img = 1:n_img
    scr = img_scr{1}(img);
    %new_scr = (scr - mf_thr) * M(num2str(q));
    new_scr = scr * M(num2str(q));

    if (isKey(S_M, num2str(img)))
      % update score
      S_M(num2str(img)) = S_M(num2str(img)) +  new_scr;
      %if (S_M(num2str(img)) > new_scr)
      %  S_M(num2str(img)) = new_scr;
      %end
    else
      S_M(num2str(img)) = new_scr;
    end
  end

  if (rem(q, 5) == 0)
    for img = 1:n_img
      % new score to assign
      new_scr = S_M(num2str(img)) / 5;

      new_scores{m_idx * 5 + 1}(img) = new_scr;
      new_scores{m_idx * 5 + 2}(img) = new_scr;
      new_scores{m_idx * 5 + 3}(img) = new_scr;
      new_scores{m_idx * 5 + 4}(img) = new_scr;
      new_scores{m_idx * 5 + 5}(img) = new_scr;
    end
    m_idx = m_idx + 1;
    % reset the Score Map (hash table)
    S_M = containers.Map;
  end
end

% original scores
% sort images and evaluate
[~, ranks] = sort (cell2mat(scores')', 'descend');
map = compute_map (ranks, gnd);
fprintf('[origin] k %d, map %.4f\n', k, map);

% new scores
% sort images and evaluate
[~, ranks] = sort (cell2mat(new_scores')', 'descend');
map = compute_map (ranks, gnd);
fprintf('[new] k %d, map %.4f\n', k, map);
