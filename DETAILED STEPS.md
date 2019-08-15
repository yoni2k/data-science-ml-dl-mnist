# Conclusions per execution:
## Conclusions 1 - initial:
### Details
- Num epochs
    - Max number of Epochs (15) with a very quick StopFunction (0.1 diff, patience 2) was reached only:
        - With Hidden Width = 10 (clearly sub-optimal)
        - With accuracy not approaching great results (range 0.8-0.9)
        - With large batch sum (1000) - probably with smaller number reached best quicker
        - Mostly with larger num layers (5) - does it mean that for smaller number gave up quicker or reached quicker?
    - Number of epochs > 10, but < 15
        - No great results .88-.95
        - Mostly hidden width = 10 (a few with hidden width 64 gave relatively good results)
        - Mostly large batch sum (1000), a few with small batch size (100) finished relatively quickly (11 epochs)
    - 3 epochs only with worst result (0.1)
    - 4-6
        - some of the best results in all categories
        - Most batch size 100 (smaller)
        - Most num layers 5 (larger)
        - Most Hidden width larger 64 (with smaller much worse results)
    - 7-10
        - Some of the best accuracy, but not best time
    - Thoughts
        - Seems that don't need more epochs
        - With given hyperparameters, could even use less epochs, but since will make the StopFunction to be slower, need to leave this number and check again if enough / too much 
- Average epoch time
    - Range 6 - 11
    - Worse with smaller batch size, better with larger batch size 
    - < 7
        - Larger batch size (1000)
        - Usually worse total times and larger number of iterations
    - \> 9
        - Small batch sizes (100)
        - Large number of layers (5)
        - Some of the best accuracies
    - Thoughts: 
        - Not a helpful metric, since has high correlation to batch sizes - could be better fewer slower epochs with smaller number of batch sizes
- Hidden width [10,64]
    - 10
        - Never gives good accuracy
        - Sometimes gives good times, but not with good accuracy
    - Thoughts: 
        - Need to enlarge, see if more than 64 is better, or less
- Batch size [100,1000]
    - 1000 - never gives best results (perhaps because of quick StopFunction)
    - Thoughts:
        - 100 seems better than 1000, need to try something in-between
- Num layers [4, 5]
    - Best accuracy results both in 4 and 5, but more in 5
    - Time-wise, 5 seems to be better 
    - Thoughts:
        - Can't give up on 4 completely
        - Some time in the future, need to try 3 to see if it's much worse than 4
        - 6 might possibly be better - need to try
- Accuracy
    - < .9
        - Large batch size (1000)
        - Small hidden width size (10)
        - A lot of sigmoid functions
        - Also usually large times
    - [.978 - .987] - best
        - All batch size 100
        - All width 64
        - Usually comes with pretty good times too
        - 4-8 epochs
        - Shortest time - 30 sec, longest - 90 sec
    - \> .95, and <.978
        - Large number of batch size 1000 !
        - All width 64
        - Usually comes with pretty good times too
        - 4-9 epochs
    - Thoughts:
        - Can dismiss width 10
        - Batch size 100 seems better, but can't completely dismiss 1000
- Train time
    - 21 - 107 seconds range
    - \> 100
        - 15 epochs
        - Batch size 1000
        - Hidden width 10
        - None of the best times
    - [84, 100]
        - Mostly large number of epochs (13-15)
        - Mostly batch size 1000
        - Mostly not great accuracies
    - 21 - didn't learn 
    - [30-34]
        - With batch size 100 and Hidden width 64
            - some of the best accuracies too and some of the best efficiencies
            - smallest number of epochs (4-5)
    -  [34-50]
        - Mostly batch size 100
        - Good results with hidden width 64
        - Mostly 4-7 epochs
        - Good ones 7-8 seconds per epoch
- Efficiency (accuracy / time)
    - 0.005 to .032
    - [0.31,0.32]
        - Batch size 100
        - Num layers varies
        - Funcs relu, tanh, [tanh]
        - Number of epochs - 4
    - [0.25, 0.26]
        - Batch size 100
        - Num layers - 5 (longer to reach than varying 4 above)
        - Hidden width - 64
        - Functions - `relu` first, but sometimes also `tanh`
    - < .01
        - Batch size 1000
        - Usually num layers = 5, and not 4
        - Hidden width = 10
        - Usually 15 epochs
        
### Conclusions going forward:
- Num layers - not clear what's better - 5 sometimes gives better results but usually slower.  Try 3 to prove than worse, try 6 to prove that worse than 5 because of time it takes
- Batch size - 100 is usually better than 1000 but not by knockout - try something in the middle, and lower than 100
- Hidden width - 64 is clearly better than 10, need to try higher and lower than 64
- Hidden funcs - 'relu' and 'tanh' should not be removed, possibly need to add 'softmax'
- ACCURACY_IMPROVEMENT_DELTA, ACCURACY_IMPROVEMENT_PATIENCE - 0.01 and 2.  Patience leave 2 since cycles take a lot of time. Better to have delta 0.001 at least,
    but will take much longer, so for brute solutions, leave 0.01
- MAX_NUM_EPOCHS - was 15, leave 15 for if doing delta 0.001, can put 10 if for now staying with delta 0.01

## Conclusions 2 - number of layers:
### Details
- Layers
    - Layer 3
        - Some of the worst accuracies (92-96)
        - didn't use all 10 epochs
        - Some of the best efficiencies because of good times
    - Layer 4
        - Not great accuracies
    - Layer 5
        - Some of the better accuracies and efficiencies
    - Layer 6
        - Mixed - sometimes best, sometimes worst in all categories
- Accuracy
    - Best - Some 4, some 5, some 6 layers, none 3
    - Top 1/3 - 5 and 6
    - Top 10% - all 6
    - 4 appears in top 40% - .011 difference from best
- Efficiency (accuracy / time)
    - Top 10% already has 6 and 5
    - Top 27% has 4 already, but not with best result    
### Conclusions going forward: 
- With the given other parameters, 5 seems best, but 4 and 6 should be left in the trials (also interesting to check if 3 and 7 for sure worse)

## Conclusions 3 - batch size: [100,500,1000]
### Details
- Accuracy
    - 100s - vast majority in top half, a few from 500 and 1000 either take longer, or not the top accuracy
- Efficiency (accuracy/time):
    - Top half are 100s with best accuracies, or 500 / 1000 not with best accuracy
### Conclusions going forward:
    - 100 seems better than alternatives, look around 100  

## Conclusions 4 - batch size: [50,100,250,500]
### Details
- Batch size 500 is never best both in accuracy and efficiency together
- Top half in Accuracy and Efficiency Efficiency (accuracy/time):
    - Batch sizes 50, 100, 250 have all some of the best results
    - 250 is not in both accuracy and time results
### Conclusions going forward:
    - Need to rerun with not such a quick StopFunction - at least 0.001, and 15 iterations
    - Can drop 500, can introduce a number between 50 and 100, and 100 and 250
    - Also noticed that my functions didn't give all options, rerunning also for that reason  


## Conclusions 5 - batch size: [50,75,100,170,250] + much slower StopFunction + all function variations
### Details
- Seems to have some overfitting on validate accuracy
- Some of the best results it wasn't enough to have 25 epochs
- Batch
    - 50 - some of the best accuracies and efficiencies
    - 75 - some of the best accuracies and efficiencies
    - 100 - same
    - 170 & 250 - some of the best accuracies, but not the most efficient
- Accuracy
    - absolutely best is with 100
    - some of the best in each one of the categories: 50,75,100,170,250
- Efficiency
    - Some of the best in all categories besides 250, but accuracies are not the best
- Accuracy product
    - Best .9945
    - Some of the best in all categories, including 250
- Accuracy product per time
    - Some of the best in all categories, besides 250, but accuracies not good
- Best both accuracy product and Accuracy Product per Time
    - Accuracy Product - around .99 (.9905-.9912) with best being .9945
### Conclusions going forward:
- Add 2 new metrics: product of accuracies and efficiency of product (product of accuracies / time) - to make sure there is no overfitting on validate accuracy
- Enlarge number of epochs to 30 since some of the best results it wasn't enough to have 25 epochs
- The whole range seems to produce some of the best results, with batch 250 possibly less so.  Perhaps run 150 as default, and 50 and 250 as extras from now on
- Next step: try different number of widths: 25, 50, 75

## Conclusions 6 - hidden width size: [25,50,75] + much slower StopFunction + all function variations
### Details
- Epochs - max 25 - Some of the best results reached maximum of 25 epochs, but also many of the bad ones
- Width 25 - some of the worst in all parameters
- Width 50 - some of the best in accuracies, but not so much efficiency 
- Width 75 - best accuracies, some of the best efficiencies
- Accuracy - Most of best width 75, some 50
- Product accuracy - best 0.9956 - vast majority 75
### Conclusions going forward:
- Width 75 seems best, need to check if going up helps (100?)
- For now setting on 75
- Next run: try if 4 layers give drastically worse results

## Conclusions 7 - 4 instead of 5 layers
### Details
- Num epochs - seems that 25 was enough
- Got 0.9964 product accuracy - so 4 layers seems more than enough
### Conclusions going forward:
- Leave 4 layers for now, play with other parameters - fine tune other parameters on 4 layers

## Conclusions 8 - 4 layers with different options of all 4 activation functions, batch_sizes = [100, 150, 200], hidden_widths = [60, 75, 100], 
### Details
- Number of epochs - 25
    - 1 of the best ones didn't finish, but most didn't need more, and it seems that have plenty similar that did
    - A third almost reached 25 epochs, so it's good we stopped at 25, otherwise it would take much longer
- Activation functions
    - softmax is not helpful ever
    - sigmoid is never first, but in some cases not a lot behind.  Can for now remove as first, since plenty similar
- Accuracies Product - best .9993 (Validate 0.9998, Train 0.9995)
    - Top 5% - .9981-.9993
        - Different batch sizes [100, 150, 200], but more 200, and where not 200, there is another similar result in 200
        - Hidden widths all largest 100
        - First activation function is always relu / tanh
    - Best: Batch size 200, Hidden funcs - ('tanh', 'relu'), Hidden width - 100 
### Conclusions going forward:
- If would continue: 
    - Width - possibly more than 100 could do better?
    - Batch size - 200 is best or same, perhaps try higher?
    - Remove softmax function altogether, remove sigmoid as first
- Need to check with endless iterations, but batch size largest.  Possibly same or better result, but longer?
- Need to check with different seed to make sure what was chosen was not luck for the specific split
- Need to check without batch sizes at all but with endless epochs to see if always get better and consistent results

# Conclusions 9 - running single function that was found to be the best numerous times with different seeds to check if it's consistent
### Details
- Ran 5 times, Accuracies Product 4/5 .995 and above, but 1/5 .98, with 1 being .9993
### Conclusions going forward:
- Check if it's a matter of batching, and whether it's better perhaps not to batch to get best results

# Conclusions 10 - running without batches at all
- Took very very long, and results (at least with current StopFunction) are very bad - accuracy 0.92
- Decided that batches are needed, even if they are not very small
### Conclusions going forward:
- Try with even slower StopFunction (0.0001 and not 0.001 delta that was used till now)
- Need to check again if reading the data, or preparing the data causes the fluctuation between results
- Try with much larger batches than 100 or 150 used till now - perhaps could have better results without paying too much in time

# Conclusions 11 - running with even slower StopFunction (0.0001 and not 0.001 delta that was used till now)
- Number of epochs needed most: 24, usually lower 
- Product accuracy - between .9946 and .9993 (average of 5: .9977)
- Training time - 2-3 minutes
### Conclusions going forward:
- Extra time / extra number of epochs done due change of StopFunction is not drastic, but results improved, although they are not exactly the same
- Previous conclusion: - Need to check again if reading the data, or preparing the data causes the fluctuation between results