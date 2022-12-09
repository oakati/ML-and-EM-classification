function [test_targets, param_struct_em] = EM(train_patterns, train_targets, test_patterns, Ngaussians, param_struct_mle, init_mode)

% Classify using the expectation-maximization algorithm
% Inputs:
% 	train_patterns	- Train patterns
%	train_targets	- Train targets
%   test_patterns   - Test  patterns
%   Ngaussians      - Number for Gaussians for each class (vector)
%
% Outputs
%	test_targets	- Predicted targets
%   param_struct_em    - A parameter structure containing the parameters of the Gaussians found

classes             = unique(train_targets); %Number of classes in targets
Nclasses            = length(classes);
Nalpha				= Ngaussians;						 %Number of Gaussians in each class
Dim                 = size(train_patterns,1);

max_iter   			= 100;
max_try             = 5;
Pw					= zeros(Nclasses,max(Ngaussians));
sigma				= zeros(Nclasses,max(Ngaussians),size(train_patterns,1),size(train_patterns,1));
mu					= zeros(Nclasses,max(Ngaussians),size(train_patterns,1));

switch init_mode
    case 'prev_mle'
        %The initial guess is based on previous MLE. If it does not converge after
        %max_iter iterations, a random guess is used.
        for i = 1:Nclasses
            for j = 1:Ngaussians(i)
                Pw(i,j)         = param_struct_mle(i).w;
                sigma(i,j,:,:)  = param_struct_mle(i).sigma;
                mu(i,j,:) = param_struct_mle(i).mu';
            end
        end
        %Randomize initial guesses.
        start               = 0.9;
        stop                = 1.1;
        Pw = distortInitialValues(start,stop,Pw);
        sigma = distortInitialValues(start,stop,sigma);
        mu = distortInitialValues(start,stop,mu);
end


%Do the EM: Estimate mean and covariance for each class
for c = 1:Nclasses
    train	   = find(train_targets == classes(c));

    if (Ngaussians(c) == 1)
        %If there is only one Gaussian, there is no need to do a whole EM procedure
        sigma(c,1,:,:)  = sqrtm(cov(train_patterns(:,train)',1));
        mu(c,1,:)       = mean(train_patterns(:,train),2)';
    else

        sigma_i         = squeeze(sigma(c,:,:,:));
%         old_sigma       = zeros(size(sigma_i)); 		%Used for the stopping criterion
%         mu         = squeeze(mu(c,:,:));
        old_mu       = zeros(size(mu)); 		%Used for the stopping criterion
        iter			= 0;									%Iteration counter
        n			 	= length(train);					%Number of training points
        qi			    = zeros(Nalpha(c),n);	   	%This will hold qi's
        P				= zeros(1,Nalpha(c));
        Ntry            = 0;

        while ((sum(abs(mu-old_mu),'all') > (1e-16)) & (Ntry < max_try)) %/(4.5097e+03^5)
            %             old_sigma = sigma_i;
            old_mu = mu;

            %E step: Compute Q(theta; theta_i)
            for t = 1:n
                data  = train_patterns(:,train(t));
                for k = 1:Nalpha(c)
                    P(k) = Pw(c,k) * p_single(data, squeeze(mu(c,k,:)), squeeze(sigma_i(k,:,:)));
                end

                for i = 1:Nalpha(c)
                    qi(i,t) = P(i) / sum(P);
                end
            end

            %M step: theta_i+1 <- argmax(Q(theta; theta_i))
            %In the implementation given here, the goal is to find the distribution of the Gaussians using
            %maximum likelihod estimation, as shown in section 10.4.2 of DHS

            %Calculating mu's
            for i = 1:Nalpha(c)
                mu(c,i,:) = sum((train_patterns(:,train).*(ones(Dim,1)*qi(i,:))),2)'/sum(qi(i,:),2)';
            end

            %Calculating sigma's
            %A bit different from the handouts, but much more efficient
            %             for i = 1:Nalpha(c)
            %                 data_vec = train_patterns(:,train);
            %                 data_vec = data_vec - squeeze(mu(c,i,:)) * ones(1,n);
            %                 data_vec = data_vec .* (ones(Dim,1) * sqrt(qi(i,:)));
            %                 sigma_i(i,:,:) = sqrt(abs(cov(data_vec',1)*n/sum(qi(i,:),2)'));
            %             end

            %Calculating alpha's (gamma)
            Pw(c,1:Ngaussians(c)) = 1/n*sum(qi,2)';

            iter = iter + 1;
            disp(['Iteration: ' num2str(iter)])
            param_struct_em(c).iter = iter;
            
            if (iter > max_iter)
                theta = randn(size(sigma_i));
                iter  = 0;
                Ntry  = Ntry + 1;

                if (Ntry > max_try)
                    disp(['Could not converge after ' num2str(Ntry-2) ' redraws. Quitting']);
                else
                    disp('Redrawing weights.')
                end
            end

        end
%         mu(c,:,:,:) = mu;
%         sigma(c,:,:,:) = sigma_i;
    end
end

%Classify test patterns
for c = 1:Nclasses
    param_struct_em(c).p       = length(find(train_targets == classes(c)))/length(train_targets);
    param_struct_em(c).mu      = squeeze(mu(c,1:Ngaussians(c),:));
    param_struct_em(c).sigma   = squeeze(sigma(c,1:Ngaussians(c),:,:));
    param_struct_em(c).w       = Pw(c,1:Ngaussians(c));
    for j = 1:Ngaussians(c)
        param_struct_em(c).type(j,:) = cellstr('Gaussian');
    end
    if (Ngaussians(c) == 1)
        param_struct_em(c).mu = param_struct_em(c).mu';
    end
end
[test_targets, p_error_given_x] = classify_paramteric(param_struct_em, test_patterns');

Uclasses = unique(test_targets);

for i = Uclasses
    indices = (test_targets == i);
    b = p_error_given_x(:,indices);
    a = max(b);
    param_struct_em(i).p_error_given_x = (sum(b)-a)./sum(b);
end

%END EM

function p = p_single(x, mu, sigma)

%Return the probability on a Gaussian probability function. Used by EM

p = 1/(2*pi*abs(det(sigma)))^(length(mu)/2)*exp(-0.5*(x-mu)'*(sigma\(x-mu)));

