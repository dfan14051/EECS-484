%computes derivatives w/rt weights for BP
function  [dWL_cum,dW_Lminus1_cum,delta_L_cum,delta_Lminus1_cum] = compute_dW_from_sensitivities(Wji,Wkj,b1_vec,b2_vec, training_patterns,targets)

temp =size(targets);
P=temp(1); %number of training patterns
K=temp(2); %number of outputs
temp = size(Wji);
J=temp(1); %number of interneurons
I=temp(2); %dimension of input patterns
delta_L_cum = zeros(K,1);
delta_L = delta_L_cum;
delta_Lminus1_cum = zeros(J,1);
delta_Lminus1 = delta_Lminus1_cum;
dWL_cum = Wkj*0; 
dWL = Wkj*0;
dW_Lminus1 = Wji*0;
dW_Lminus1_cum = Wji*0;

for p=1:P %make the P loop the outer loop, since need to re-use results of
    %network simulation for pattern p many times
    stim_vec= training_patterns(p,:)';
    %need to compute outputs of both j and k layers for stimulus pattern p
    [outputj,outputk]=eval_2layer_fdfwdnet(Wji,Wkj,b1_vec,b2_vec,stim_vec);
    err_vec = outputk - targets(p,:);

    phi_prime_L_vec = logsig(Wkj*outputj + b2_vec).*(1-logsig(Wkj*outputj + b2_vec)); %FIX ME!!
    delta_L = phi_prime_L_vec.*err_vec; %FIX ME!!
    delta_L_cum=delta_L_cum + delta_L;

    dWL = outputj*delta_L'; %FIX ME!!
    dWL_cum = dWL_cum+dWL;

    phi_prime_Lminus1_vec = logsig(Wji*stim_vec + b1_vec).*(1-logsig(Wji*stim_vec + b1_vec));  %FIX ME!!
    delta_Lminus1 = Wkj'*delta_L.*phi_prime_Lminus1_vec; %FIX ME!!
    delta_Lminus1_cum = delta_Lminus1_cum + delta_Lminus1;

    dW_Lminus1 =  stim_vec*delta_Lminus1'; %FIX ME!!
    dW_Lminus1_cum = dW_Lminus1_cum+dW_Lminus1';
    
end %done evaluating influence of all P stimulus patterns
%delta_L_cum
%delta_Lminus1_cum
%dWL_cum
%dW_Lminus1_cum
