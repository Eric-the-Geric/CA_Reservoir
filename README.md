# CA_Reservoir
Reservoir computing using different types of cellular automata:
- Only Elementary CA and game of life have been implimented

# Seems like image classification doesn't reallly work so well. The only examples I have seen in literature using CA are the n-bit memory tasks


But this might be useful for those people who don't know how to set up a reservoir system in the first place. It works by parsing input through a dynamical systen, like a cellular automata, and then after N iterations connect the output to a fully-connected MLP. The intuition is it reduces training compute because you only need to train the final layer but for a simple task like MNIST you only need a small MLP in the first place and in fact applying GOL makes it perform worse. Also the more iterations of GOL you run, the wrose the performance (accuracy + precision). Need to find a new new task for a system like this... work in progresss..
