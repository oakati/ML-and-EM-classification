%% some initializations
% Author: Ömer Katı
clear,clc; format shortE;format compact;
dataset_example = strcat(pwd,"\");
classes = ["chair", "desk", "sofa", "table", "toilet"];
c = size(classes,2);    % #classes
d = 400;                % #features
d_p = 61;               % reduced # features
card_D = 100*ones(1,c); % cardinality of D sets
card_T = 25*ones(1,c);  % cardinality of T sets

D = zeros(card_D(1),d,c);           % training subsets
T = zeros(card_T(1),d,c);           % test subsets
D_targets = zeros(card_D(1),1,c);    % training targets
T_targets = zeros(card_T(1),1,c);    % test targets

prior = 0.2*ones(1,c); % p(w_c) = card_D(i)/sum(card_D)
clearvars card_D card_T;
%% reading datasets
for i = 1:c
    D(:,:,i) = readmatrix(strcat(dataset_example,classes(i),"\train.txt"));
    D_targets(:,:,i) = i;
    T(:,:,i) = readmatrix(strcat(dataset_example,classes(i),"\test.txt"));
    T_targets(:,:,i) = i;
end
clearvars dataset_example i data_dir classes
%% merging datasets
D_total = [D(:,:,1);D(:,:,2);D(:,:,3);D(:,:,4);D(:,:,5)]; % whole training set
T_total = [T(:,:,1);T(:,:,2);T(:,:,3);T(:,:,4);T(:,:,5)]; % whole test set
D_targets_total = [D_targets(:,:,1);D_targets(:,:,2);D_targets(:,:,3);D_targets(:,:,4);D_targets(:,:,5)]; % whole training set
T_targets_total = [T_targets(:,:,1);T_targets(:,:,2);T_targets(:,:,3);T_targets(:,:,4);T_targets(:,:,5)]; % whole test set
clearvars D D_targets T T_targets
%% principle component analysis (PCA)
[D_total_pca, D_targets_total, UW, mu, W] = PCA(D_total', D_targets_total, d_p);
D_total_pca = D_total_pca';
T_total_pca = (W*T_total')';
clearvars mu UW W
%% multiple discriminant analysis (MDA)
[D_total_mda, D_targets_total, w, J_W] = MultipleDiscriminantAnalysis(D_total_pca', D_targets_total, c);
D_total_mda = D_total_mda';
T_total_mda = (w'*T_total_pca')';
clearvars J_W w
%% MLE
[predictions_mle, param_struct_mle] = ML(D_total_mda', D_targets_total, T_total_mda, T_targets_total);
%% p_error_mle
p_error_mle = 0;
for i = 1:c
    p_error_mle = p_error_mle + mean(param_struct_mle(i).p_error_given_x);
end
p_error_mle = p_error_mle/c;
%% recall_mle and precision_mle
tp_plus_fp_mle = zeros(1,c);
tp_plus_fn_mle = zeros(1,c);
recall_mle = zeros(1,c);
precision_mle = zeros(1,c);
for i = 1:c
    tp_plus_fp_mle(i) = sum((predictions_mle' == i));
    tp_plus_fn_mle(i) = sum((T_targets_total == i));
    recall_mle(i) = sum((predictions_mle' == i) & (T_targets_total == i))/tp_plus_fn_mle(i);
    precision_mle(i) = sum((predictions_mle' == i) & (T_targets_total == i))/tp_plus_fp_mle(i);
end
accuracy_mle = sum((predictions_mle' == T_targets_total))/length(T_targets_total);

%% EM
Ngaussians = [2, 2, 2, 2, 2];
init_mode = 'prev_mle';
[predictions_em, param_struct_em] = EM(D_total_mda', D_targets_total', T_total_mda, Ngaussians, param_struct_mle, init_mode);
%% p_error_em
p_error_em = 0;
for i = 1:c
    p_error_em = p_error_em + mean(param_struct_em(i).p_error_given_x);
end
p_error_em = p_error_em/c;
%% recall_em and precision_em
tp_plus_fp_em = zeros(1,c);
tp_plus_fn_em = zeros(1,c);
recall_em = zeros(1,c);
precision_em = zeros(1,c);
for i = 1:c
    tp_plus_fp_em(i) = sum((predictions_em' == i));
    tp_plus_fn_em(i) = sum((T_targets_total == i));
    recall_em(i) = sum((predictions_em' == i) & (T_targets_total == i))/tp_plus_fn_em(i);
    precision_em(i) = sum((predictions_em' == i) & (T_targets_total == i))/tp_plus_fp_em(i);
end
accuracy_em = sum((predictions_em' == T_targets_total))/length(T_targets_total);
clearvars i;
