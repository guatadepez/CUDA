function fitness = fitness_cond(front,A,b,w)

  % w es de 1X289, debe ser de 289x1
  N = length(b); %289
  E= 0;
  for i=1:N
      producto = 0;
      for j=1:N
         producto = producto + A(i,j)*w(j); 
      end
      E = E + abs(producto-b(i));
  end
  rmse = sqrt(E/N);

  F0 = rmse; % raiz okk

  if N == 289 || N == 1089
      %sum nodos <= sumFuente
    if sum(w) <= sum(b)
        F1 = 0;
    else
        F1 = 30; %se castiga los q no cumplen la cond
  end


   nodos = size(A,1);
   fronteras = sort(front); % primera fila estan los nodos frontera, 1x64
   cant_front = length(fronteras);
   f2 = rand(cant_front,1); % vector de 64X1 para 289 nodos

    %condicion borde 0, 
    i=1;
    for nodo=1:nodos
      if find(fronteras == nodo) % el nodo esta en la frontera
        if w(nodo) == 0
          f2(i) = 0;
        else if w(nodo) <= 100
          f2(i) = 5;
        else if w(nodo) <= 1000
          f2(i) = 20;
        else
          f2(i) = 50; %se castiga a los w cuyo valor en los nodos esta alejado de 0
        end
      end

      i = i+1;
      if i > cant_front
        break;
      end
    end  
  end
  F2 = sum(f2);

  if N == 289 || N == 1089
    fitness = F0 + F1 + F2;
  else
    fitness = F0;
  end

fitness
end