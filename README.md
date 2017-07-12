# Neural

About
-----
This Program allows the user to create various neural networks by varying the ammount of layers and neurons.The 
program by itself is not useful, it is intended to be used as a platform for other projects.The network structure
is stored in a 3 dimensional array neuron[x,y,z] where x is the layer, y is the neuron from the top, and z is the
input or output. For example to read the output of the third neuron in the second layer --->> **double out = neuron[1,2,1];**
The activation function used is the Sigmoid function: S(x) = 1/(1+Math.Pow(Math.E, -x)).

Usage
-----
The directory is a visual studio solution. You could open  the .sln file to edit the program or run the exe
in bin/Debug with default input and targets.
