import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

"""
Talvez não tenha percebido que o algoritmo perceptron se assemelha e muito ao gradiente descendente estocástico.
Na realidade, a classe Perceptron do Scikit-Learn é equivalente ao uso de um SGDClassifier com os seguintes 
hiperparâmetros: loss="perceptron", learning_rate="constant", eta0=1 (taxa de aprendizagem) e penalty=None (sem regularização).

Os perceptrons têm uma série de deficiências graves -  em especial, o fato de serem incapazes de resolver alguns 
problemnas corriqueiros (e.g., o problema de classificação Exclusive OR (XOR). Isso vale para qualquer outro modelo 
de classificação linear (como os classificadores de regressão logística); no entanto, os investigadores esperavam mais dos 
perceptrons, e alguns ficaram tão deceptionados que deixaram de lado as rede neuronais em favor de problemas de nível 
superior, como lógica, resolução de problemas e mecanismos de procura. Acontece que algumas das limitações das perceptrons podem 
ser eliminadas ao empilhar diversas perceptrons. A RNA resultante desse processo chama-se perceptron multicamada (MLP).  

Perceptron multicamada e retropropagação

Uma MLP é composta de uma camada de entrada (passagem), uma ou mais camadas de TLUs (Unidade Lógica de Limiar), chamadas 
de camadas ocultas, e uma camada final TLUs, chamda de saída. As camadas próximas à camada de entrada normalmente chamam-se camadas inferiores e as próximas às saídas 
chamam-se camadas superiores. Cada camada, tirando a camada de saída, inclui um neurónio de viés e está totalmente conectada à próxima camada.

PS: O sinal circula apenas numa direção (das entradas às saídas), logo, essa arquitetura é um exemplo de uma rede neuronal feedforward (FNN).

Algoritmo de treino de retropropagação (backpropagation): Em síntese, é o gradiente descendente usando uma técnica eficiente para 
calcular automaticamente os gradientes, bastam duas passagens pela rede (um forward pass e um backward pass), que o algoritmo 
de retropropagação consegue calcular o gradiente do erro da rede em relação a cada parâmetro de modelo único. Por outras palavras, ele 
pode identificar como cada peso de conexão e cada viés devem ser ajustados para reduzir o erro. Tendo esses gradientes, ele somente executa 
uma etapa regular do gradiente descendente e todo o processo é repetido até a rede convergir para a solução.

PS: O cálculo automático dos gradientes chama-se diferenciação automática ou autodiff. Existem técnicas de autodiff, e cada 
uma delas tem as suas vantagens e desvantagens. A técnica empregada pela retropropagação chama-se autodiff de modo reverso 
[reverse-mode autodiff]. É rápida, precisa e adapta-se bem quando a função para diferenciar apresenta muitas variáveis (por exemplo, 
pesos de conexão) e poucas saídas (por exemplo, uma perda).

Vamos analisar o algoritmo:
1.  Ele manipula um mini-batch por vez (por exemplo, contendo 32 instâncias cada) e realiza o treino completo diversas vezes. 
    Cada passagem é chamada de época.
2.  Cada mini-batch é passado para a camada de entrada da rede, que a envia para a primeira camada oculta. O algoritmo calcula 
    a saída de todos os neurónios nesta camada (para todas as instâncias do mini-batch). O resultado é passado para a próxima camada, 
    a sua saída é cálculada e passada para a próxima camada, e assim sucessivamente até obtermos a saída da última camada, a camada de saída.
    É chamado forward pass: é justamente como efetuar as predições, exceto que todos os resultados intermédios são preservados, pois são 
    necessários para o backward pass.
3.  Em seguida, o algoritmo calcula o erro de saída da rede (ou seja, usa uma função de perda que compara a saída desejada e a saída real da 
    rede e retorna alguma medida de erro).
4.  Depois, ele calcula o quanto cada conexão de saída contribuiu para o erro. Esse processo é feito analiticamente aplicando a 
    regra da cadeia [chain rule] (talvez a regra mais fundamental no cálculo), o que torna essa etapa rápida e precisa.
5.  O algoritmo calcula quanto dessas contribuições de erro veio de cada conexão na camada inferior, usando mais uma vez a regra da cadeia, 
    trabalhando em backward até o algoritmo atingir a camada de entrada. Conforme explicado anteriormente, essa passagem reversa computa com 
    eficiência o gradiente de erro em todos os pesos de conexão na rede, retropropagação o gradiente de erro (daí o nome do algoritmo).
6.  Por último, o algoritmo executa uma etapa do gradiente descendente para ajustar todos os pesos de conexão na rede, utilizando os 
    gradientes de erro que acabou de calcular.

Resumidamente: o algoritmo de retropropagação primeiro faz uma predição (forward pass) e calcula o erro, depois passa por cada camada no sentido 
inverso a fim de calcular a contribuição do erro de cada conexão (reverse pass) e, por fim, ajusta os pesos da conexão para reduzir o erro (etapa do 
gradiente descendente).

É importante iniciar todos os pesos de conexão das camadas ocultas aleatoriamente ou o treino irá por água a baixo. Por exemplo, se iniciar os pesos 
e vieses para zero, todos os neurónios numa determinada camada serão iguais, assim a retropropagação os afetará exatamente da mesma forma, de modo 
que eles permaneçam idênticos. De grosso modo, apesar de ter centenas de neurónios por camada, o seu modelo comportar-se-á como se tivesse apenas 
um neurónio por camada: ele não será muito inteligente. Agora, caso inicie aleatoriamente os pesos, quebra a simetria e permite que a retropropagação 
treina um conjunto diversificado de neurónios.

Para que este algoritmo funcione adequadamente, os seus autores fizeram uma alteração imprescindível na arquitetura da MLP: eles substituíram a função 
degrau pela função logística (sigmoide). Isto foi essencial porque a função degrau contém apenas segmentos planos; deste modo, não existe gradiente com o
qual se trabalhar (o gradiente descendente não pode deslocar-se numa superfície plana), ao passo que a função logística comporta uma derivada diferente de zero 
bem definida em todos os lugares, possibilitando que o gradiente descendente progrida a cada etapa. De facto, o algortimo de retropropagação funciona bem 
com outras funções de ativação, não somente com a função logística. 

Exemplos: Função tangente hiperbólica (tanh(z)); Função de unidade linear retificada (ReLU(z))
"""

iris = load_iris()
X = iris.data[:,(2,3)] # comprimento da pétala, largura da pétala
y = (iris.target == 0).astype(np.int32) # Iris setosa?

per_clf = Perceptron()
per_clf.fit(X,y)
y_pred = per_clf.predict([[2,0.5]])