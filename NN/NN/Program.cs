using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN
{
    class Program
    {
        public static Random rnd = new Random();
        public static double learningrate = 1; //you can modify this and find the best value for your network
        public static double[,,] neuron;
        public static double[] weights, target, output;
        public static int[] aon;
        public static int hw, biggest;
        static void Main(string[] args)
        {

            //this part sets up arrays for the structure of the network

            Console.Write("Total layers >> ");
            int hl = Convert.ToInt16(Console.ReadLine());
            aon = new int[hl];
            Console.WriteLine("Input layer sizes");
            for (int k = 0; k < hl; k++)
            {
                Console.Write("Layer " + (k+1).ToString() + " >> ");
                aon[k] = Convert.ToInt16(Console.ReadLine());
                if (aon[k] > biggest)
                {
                    biggest = aon[k];
                }
            }
            neuron = new double[hl, biggest, 2];
            target = new double[aon[aon.Length - 1]];
            hw = 0;
            for (int w = 0; w < aon.Length - 1; w++)
            {
                hw += (aon[w] * aon[w + 1]);
            }
            int biases = 0;
            for (int bs = 1; bs < aon.Length; bs++)
            {
                biases += aon[bs];
            }
            weights = new double[hw + biases];
            Console.WriteLine("You have " + weights.Length + " weights");
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = uniformrandom(3);
            }

            /* Below is an example for prosessing 2 input neurons and
               learn from 2 targets with Backpropagation. You can add
               targets and input neurons but remember to input the
               sizes of the layers correctly at the beginning
            */

            Console.Write("Iterations >> ");
            long oo = Convert.ToInt64(Console.ReadLine());
            for (long i = 0; i < oo; i++)
            {
                int x = rnd.Next(0,3);
			    if (x == 0){
				    neuron[0, 0, 1] = -1;
				    neuron[0, 1, 1] = 1;
				    target[0] = 0;
                    target[1] = 1;
			    }
			    else if (x == 1){
				    neuron[0, 0, 1] = -0.7;
				    neuron[0, 1, 1] = 0.7;
				    target[0] = 1;
                    target[1] = 0;
			    }
                else if (x == 2)
                {
                    neuron[0, 0, 1] = 0.5;
                    neuron[0, 1, 1] = -0.5;
                    target[0] = 0.5;
                    target[1] = 0;
                }
			    ForwardPass(); // Get output
			    BackPropagation(); //Fix weights
            }

            Console.WriteLine("------------Weights-----------");
            foreach (double x in weights){
			    Console.WriteLine(x);
		    }
            Console.WriteLine("----------------------------");
            Console.WriteLine("------------Network structure------------");
		    for (int an = 0;an < aon.Length;an++)
            {
                if (an == 0)
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                }
                else if (an == aon.Length - 1)
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                }
                for (int lyr = 0; lyr < aon[an]; lyr++)
                {
                    Console.Write("O ");
                }
                Console.ForegroundColor = ConsoleColor.Gray;
                Console.WriteLine();
		    }
            Console.WriteLine("----------------------------");
		    while (true){
			    for (int i = 0;i < aon[0];i++){
				    Console.Write("input["+(i)+"] >> ");
                    neuron[0, i, 1] = Convert.ToDouble(Console.ReadLine());
			    }
			    double[] think = ForwardPass();
                Console.WriteLine("-------------------");
			    foreach (double tput in think)
			    {
				    Console.WriteLine(">> I think of " + tput);
			    }
                Console.WriteLine("-------------------");
		    }
        }
        public static double[] ForwardPass()
        {
            for (int lay = 1; lay < aon.Length; lay++)
            {
                for (int i = 0; i < aon[lay]; i++)
                {
                    double ttal = 0;
                    for (int j = 0; j < aon[lay - 1]; j++)
                    {
                        int wn = 0;
                        for (int id = 0; id < lay - 1; id++)
                        {
                            wn += aon[id] * aon[id + 1];
                        }
                        wn += (j * aon[lay]) + i;
                        ttal += neuron[lay - 1,j,1] * weights[wn];
                    }
                    int add = 0;
                    for (int nub = 1; nub < lay; nub++)
                    {
                        add += aon[nub];
                    }
                    neuron[lay,i,0] = ttal + (1 * weights[hw + add + i]);
                    neuron[lay,i,1] = SigmoidFunction(neuron[lay,i,0]);
                }
            }
            output = new double[aon[aon.Length - 1]];
            for (int ou = 0; ou < output.Length; ou++)
            {
                output[ou] = neuron[aon.Length - 1,ou,1];
            }
            return output;
        }
        public static void BackPropagation(){
		    double[,] deltsums = new double[aon.Length - 1,biggest];
		    for (int i = 0;i < aon[aon.Length - 1];i++)
		    {
			    deltsums[0,i] = SigmoidFunctiondev(neuron[aon.Length - 1,i,0]) * (target[i] - output[i]);
		    }
		    for (int lay = 1;lay < (aon.Length - 1);lay++){
			    for (int j = 0;j < aon[aon.Length - lay - 1];j++){
				    double ttal = 0;
				    for (int h = 0;h < aon[aon.Length - lay];h++){
					    int ws = 0;
					    for (int l = 0;l < aon.Length - lay - 1;l++){
						    ws += aon[l] * aon[l+1];
					    }
					    ws += j*aon[aon.Length - lay] + h;
					    ttal += deltsums[lay - 1,h] * weights[ws];
				    }
				    deltsums[lay,j] = ttal * SigmoidFunctiondev(neuron[aon.Length - lay - 1,j,0]);
			    }
		    }
		    for (int lay = 0;lay < aon.Length - 1;lay++){
			    for (int j = 0;j < aon[lay];j++){
				    for (int h = 0;h < aon[lay + 1];h++){
					    int wn = 0;
					    for (int i = 0;i < lay;i++){
						    wn += aon[i] * aon[i+1];
					    }
					    wn += j*aon[lay + 1] + h;
					    weights[wn] += (deltsums[aon.Length - lay - 2,h] * neuron[lay,j,1]) * learningrate;
				    }
			    }
		    }
		    int wnn = hw;
		    for (int g = 1;g < aon.Length;g++){
			    for (int j = 0;j < aon[g];j++){
				    weights[wnn] += (deltsums[aon.Length - 1 - g,j]/1) * learningrate;
				    wnn++;
			    }
		    }
	    }
        public static double SigmoidFunctiondev(double x)// This is the inverse of the activation function
        {
            return (SigmoidFunction(x) * (1 - SigmoidFunction(x)));
        }
        public static double SigmoidFunction(double x)// The activation function used here is the sigmoid function
        {
            return 1 / (1 + Math.Pow(Math.E, -1 * x));
        }
        public static double uniformrandom(int u)
        {
            double x = 0;
            for (int i = 0; i < u; i++)
            {
                x += (rnd.NextDouble() * 2) - 1;
            }
            x = x / 3;
            return x;
        }
    }
}
