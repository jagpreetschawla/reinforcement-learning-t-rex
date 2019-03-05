# Outline
This project trains a basic NN to play chrome's `T-rex` game (the game that is displayed on the offline page of chrome browser). The aim of the project was to take a simple Neural Network and supervised training method (backpropagation using Adam) and experiment with applying it for a reinforcement learning task. No specific Reinforcement learning theory or algorithm is used in this project. It simply uses a NN trained using Adam Optimizer. Since we don't know the expected output for the NN, some basic heuristics are used to guide the network in the right direction.

![Screenshots from game played by NN](http://www.cylopsis.com/img/t-rex-snippet.png)
*Snippets from the game played by NN. It figured out that it needs to jump over cactus and birds and even learned that ducking is enough to avoid high flying birds.*

For more details than mentioned here, checkout my [blog post][post].

# To run the code
My program uses the following dependencies:

 - TensorFlow: For creating and training NN. (I used keras APIs)
 - PyAutoGUI: To take screenshots (for input to NN) and interact with the game using keyboard and mouse.
 - Numpy: For matrix operations.
 - OpenCV: For template matching (to find the game on the screen and detecting that game is over) and for basic screenshot image cleaning.

You can either install these manually or use the requirments.txt file in a virtual env.

The program will launch a browser and open the `T-rex` game. The game is cloned from [this][game] repo. If the program is unable to find a browser, you can pass the browser's path a parameter to the program. If the browser can't find the file, make sure that the git sub-repository at `t-rex-runner-gh-pages` folder is initialized and ready.

> **Note** The code uses template matching to find and detect game over. Template will matching on screen resolution and browser window size. If it fails, please change the `dino.png` and `dino_game_over.png` with new screenshots from your computer using browser window size that you expect when the code runs.

Once the page opens, the program will start controlling mouse and keyboard after a few seconds to play the game. Please don't touch the keyboard or mouse at this time to make sure that the program doesn't input in the wrong window.

> **Note:** The program controls the mouse and keyboard, so be careful. Close all important files before running because the program will keep on pressing keys even if by mistake you switch to some other window. This also makes it difficult to kill the program, you will need to quickly shift to the terminal window and press CTRL + C or whatever key is accepted by your terminal to stop the running script.

# Algorithm details
The algorithm uses a single hidden layer fully connected feedforward NN. Adam optimizer is used to train the network. It takes an image of the game as input, and outputs 3 probabilities, which are probabilities of each of 3 possible moves: jump, duck, do nothing (keep running).

## Activation functions
The hidden layer uses `tanh` activation and output layer uses `softmax`. Whichever output has the maximum probability is the move made by the model.

## Training
The training takes place after every 2 games. For every game, we are storing the inputs and move made. These gathered data will then be used for training the network using the exact same method that we use for supervised learnings. The only difference is that we don't know the exact outputs, so we use some heuristics.

The algorithm saves up to last 3000 moves of every game. At the end of the game, the last few frames are held responsible for the death of our t-rex as t-rex was doing well just before the last few moves.

Now we reinforce the network to keep doing what it was doing before these last few moves because that's what kept him alive. So we just take the predicted output (whichever had the max probability) and reinforce it by training network with this as expected output but with high probability (I used 0.9, more details are mentioned [here][heuristics]).

Handling the last few moves was more tricky. Here we know what should not be the output, but don't know about the correct output. So we should just focus on discouraging these moves and hope that this with correct moves reinforcement will handle these cases. So for these last few moves, we discourage the network by training it with these moves probability to be close to zero. The remaining probability is split among remaining moves as we don't know which of these was the right one.

## Loss function
Since we don't know the exact output, I didn't use log loss which imposes a high penalty for wrong probabilities. Instead, I used a simple mean square error, this way if our heuristic method is wrong for some inputs, we are not imposing a very high penalty.

**Read my [blog post][post] for more details.**

[post]: http://www.cylopsis.com/post/reinforcement-learning/t-rex-game/
[game]: http://wayou.github.io/t-rex-runner/
[heuristics]: http://www.cylopsis.com/post/reinforcement-learning/t-rex-game/#heuristic
