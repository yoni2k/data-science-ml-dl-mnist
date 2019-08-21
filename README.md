# MNIST with Deep Learning
- Goal: Find a good model to learn a well-known Mnist dataset of hand-written numbers
- Details: try many models with many different hyperparameters to be able to find the best model
- **Result: Found a model with Train and Validate accuracy of > 99.99% and Test accuracy ~ 99.85%**
- Inputs: 
    - Each image is 28x28 pixels, or 784 pixels
    - Each pixel is greyscale - 0 (black) to 255 (white)
- 4 layers best model
      'Batch size': 450,
      'Hidden funcs': ('relu', 'relu'),
      'Hidden width': 450
- 4 layers second best:
       'Batch size': 200,
      'Hidden funcs': ('tanh', 'relu'),
      'Hidden width': 200
- 5 layers best model:
      'Batch size': 450,
      'Hidden funcs': ('tanh', 'relu', 'tanh'),
      'Hidden width': 450
- 5 layers second best:
      'Batch size': 300,
      'Hidden funcs': ('relu', 'tanh', 'sigmoid'),
      'Hidden width': 300
                      
## Details of steps of investigation
See `DETAILED STEPS.md` for all steps and trials taken in quest to find the best model

## Hyperparameters:
### List
- **Num Loops per model** - number of loops to do per model, to see if the same hyperparameters give consitent outputs 
- **Validate loss improvement delta** - what's considered an improvement of `val_loss` (Loss of Validate set) for EarlyStopping. If this improvement is not reached after **Validate loss improvement patience** epochs below, the model is stopped 
- **Validate loss improvement patience** - number of epochs EarlyStopping waits for improvement before stopping 
- **Restore best weights** - whether EarlyStopping takes the best weights seen previously (when `val_loss` was minimal), or just take the last weights
- **Max num epochs** - in addition to EarlyStopping, whether to stop after a number of Epochs.  Because of the nature of the model, and greatly different number of epochs needed depending on many other hyperparameters, used EarlyStopping to stop, and put a very big number here (1000) not to have an effect   
- **Batch size** - size of batching while learning 
- **Num layers** - total number of layers (including input and output layers)
- **Hidden funcs** - Activation functions of hidden layers.  Try to put each function in the list in each one of the hidden layers. Input layer doesn't have an activation function, and output layer always set to `softmax` due to nature of the problem that we want percentages as output  
- **Hidden width** - width of hidden layers.
- **Learning rate** - Learning rate of the model
- **Shuffle seed** - since shuffling is done on the set before training, and we want to compare apples and apples, allow giving explicit seed to get more consistent results for comparison
### Categories
- That effect the model itself:
    - **Num layers**
    - **Hidden width**
    - **Batch size**
    - **Hidden funcs**
- That effect how the model is trained:
    - **Validate loss improvement delta** 
    - **Validate loss improvement patience**
    - **Restore best weights**
    - **Learning rate**
    
## Outputs
- see `output` folder with folders of outputs for each one of the steps of the investigation.  For detailed conclusions about each one of the steps, see `DETAILED STEPS.md`
- Output from trying various models:
    - `hyperparams.xlsx` - inputs / hyperparameters used for the run (either set for all models, or all options for different models of this run)*[]: 
    - `full.xlsx` - full results with all the inputs and outputs for each one of the models in the current run
        - Inputs - see **hyperparameters**
        - Outputs:
            - **Test Accuracy**
            - **Test Loss**
            - **Train Time** - total time took for this specific model
            - **Test Loss * Time** / **Test Loss Efficiency** - (product of **Test Loss** and **Train time**) - if 2 models give similar Test Loss, but one much quicker, it will get a higher score here 
            - **Last Validate Accuracy** - this and following 3 parameters are results of the **last** epoch, but weights are usually taken from **best** epoch
            - **Last Validate Loss**
            - **Last Train Accuracy**
            - **Last Train Loss**
            - **Num epochs** - that were actually done before the StopFunction stopped the mode
            - **Average epoch time** - time per epoch 
    - `best.xlsx` - best models from the current run based on the following parameters:
        - Best (largest) **Test Accuracy**
        - Best (smallest) **Test Loss**
        - Best (smallest) **Test Loss Efficiency** 

## General conclusions / gotchas
- In order to find the best model, the hyperparameters that don't affect the model, but how it's learning should be best.  That is very tricky, since a lot of parameters depend on each other, and it's hard to find a value that will work in all different cases.  Therefore:
    - I abandoned the maximum number of epochs early in the process, since depending on other hyperparameters, drastically
    - Started using EarlyStopping on val_loss.  Played with different configurations (delta, patience, and restore best weights).  The challenge is not to abort too early, but also not to continue needlessly.  Best values found were patience 10, delta 0.0001 for val_loss, and return to best seen weights.
        - Using larger delta causes early abandonment, smaller needless continuation
        - Using small patiences causes early abandonment, larger needed continuation
        - Not going back to best weights caused stopping the model in a bad state and getting worse values
        - Using stopping with val_accuracy wasn't good since reached 1.0000 accuracy fast   
    - Learning rate - higher than default 0.001 usually gave worse results (the minimum was missed), and lower took longer, and didn't give better results (often worse)
- Seems that there is no issue so much of overfitting - didn't see that allowing the model to run for more epochs causes overfitting.  Running less caused worse results.  Running more usually means more time, and at some point it doesn't help, but didn't see overfitting.
- Actual model settings: 
    - Number of layers
        - Seems that 5 layers gives very slightly better results than 4 layers, but not by much (~.0001-.0002) test accuracy, but takes slightly longer and less predictable (more fluctuation)
        - 3 layers had worst results in the beginning of research, didn't recheck later. Possibly need rechecking.
        - 6 layers didn't have better results in the beginning of research, and much longer to run (also many more possibilities of activation functions).  Since 4 and 5 behave close, pretty safe to assume 6 is not much better.  Possibly needs double checking.
    - Batch size - range of better batch sizes is very large - some give better accuracies with smaller, and some with larger batch sizes.  
        - Perhaps the reason for large batch size giving worse numbers is because EarlyStopping stops too early
        - No conclusion what's better, usually range of 200-500 gave better results
    - Width - at least 100-200.  Usually 300-500 gave better results.  Going all the way to 784 (number of inputs) didn't necessary give better results.
    - Activation functions
        - `softmax` - usually didn't help (gave one of the better results very rarely, and when that happened, putting a different function instead usually gave similar results) 
        - `sigmoid` - sometimes was good as the last hidden layer function.  But sometimes slow, and not extremely stable in results
        - `relu` and `tanh` are the best in different places. Most models are combinations of these 2 or 1 of them in different places - didn't see that one of them needs to be earlier/later in the model
        - The initial assumption that need as many different functions didn't prove to be correct, some of the best solutions had the same function repeated twice or even 3 times
- Notes about outputs / ways to check how good the model is:
    - It's not very hard given time and decent model and enough epochs to reach Validate and Train accuracies of 1.0000.  However, there is a gap (that I don't know to explain) between accuracies of training/validate and test accuracy.  It's not surprizing for training accuracy, since it could be explained by overfitting that causes accuracy of training to be higher, but it's not clear at all why testing and validate are so different.
    - Comments about main outputs:
        - **Test Accuracy** - main parameter looked at, best models were around 0.984-.986.  Not extemely stable, same hyperparameters on different machines / runs cause differences of .001 - .002 of even greater.
        - **Test Loss** - lowest got to was .006 / 0.007.  Lower loss usually goes together with higher accuracy, but the relationship is not completely linear. Sometimes 2 model 1 have higher accuracy, and 2nd lower loss
        - **Train Time** - varies greatly between different hyperparameters, but even runs of same hyperparameters.
        - **Test Loss * Time** / **Test Loss Efficiency** - needs to be compared on similar machine in similar conclusions. Not useful for comparison between PC and AWS runs 
    - Other outputs:
        - **Last Validate Accuracy** - some of best models reach 1.0000
        - **Last Validate Loss** - some of the best models reach 0.0000
        - **Last Train Accuracy** - some of best models reach 1.0000
        - **Last Train Loss** - some of the best models reach 0.0000
        - **Num epochs** - not extremely helpful, since depending on learning rate and batch sizes, epochs might be faster/slower, more/less improvement
        - **Average epoch time** - depends on the machine and hyperparameters, most of the times 5-12 seconds 


## Implementation details
- See `mnist.py`

## Open questions
- How come even when working with reading the data once, and using the same seed for shuffling, we still get different results when running the model a few times
- How come test accuracy is at most ~0.985 while validate accuracy reaches 1.0000
