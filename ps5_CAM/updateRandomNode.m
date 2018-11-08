   % fnc to correct nodes in matrix memory
   %YOU MUST WRITE THIS FUNCTION
   function [vec,delta] = updateRandomNode(MatrixMem,vec)
     idx = randi(size(vec, 1));
     if vec(idx) ~= sign(MatrixMem(idx,:) * vec)
         delta=1;
         vec(idx) = sign(MatrixMem(idx,:) * vec);
     else
        delta=0; %count how many nodes change their values
     end
     
    
