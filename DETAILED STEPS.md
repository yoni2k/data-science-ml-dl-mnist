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

# Conclusions 12 - reading the data moved to be performed every time in the loop, see if it makes differences larger (if data read from tensorflow_datasets different / different order every time)
- Accuracies Product - between .9976 and .9999, so it didn't help to move getting data inside the loop to get more different results
- Accuracies Product - one of them was 99.99! - which means making a mistake on 7 out of 70,000 results
### Conclusions going forward:
- Previous conclusion: Try with much larger batches than 100 or 150 used till now - perhaps could have better results without paying too much in time

# Conclusions 13 - working with batch number 500 and not 200 as till now - see if makes time worse/better, and if results stay the same
- Accuracies Product - 4/5 with .9999, and one with .9967 (and even that one with Validate Best of .9988 (.999 rounded) and Accuracy Train .9979 (rounded .998)
- Accuracies Validate - 4/5 with 1.0000! and Accuracy Train .9998 or .9999 (rounded to 1.000)
- Number of epochs - 22 to 33, for 4 better models around 32 on average
- Time - 3 minutes for the worse model, and ~3.5 minutes for the better ones - worse it to get a bit higher accuracy
### Conclusions going forward:
- Try with batch size 1000

# Conclusions 14 - batch size of 1000 instead of 500 done before
- Good but not as good results - accuracies products around .999 but not .9999 as before
### Conclusions going forward:
- Try with 500, and then go back to 200 if 500 also not as good

# Conclusions 15 - batch size of 500 instead of 200 done before
- Good but not as good results - accuracies products around .999 but not .9999 as before
- Times - around same as 200
### Conclusions going forward:
- Go back to 200 batch size

# Conclusions 16 - adding different learning rates.
Default in tf.keras.optimizers.Adam is 0.001, tried with 0.0001
- Took longer - 400-600 seconds (vs 200-300 with 0.001)
- Many more epochs - 55-80
- Accuracy was not great - product .98-.99 (stop function stops before reached the best result?)
### Conclusions going forward:
- Try with larger rate to see if will give same results as 0.001 but quicker.  Try 0.02 as suggested in the lecture

# Conclusions 17 - Learning rate 0.02
- Results - accuracies are not good - product .92%
### Conclusions going forward:
- Go back to much lower learning rates - try 0.005

# Conclusions 18 - Learning rate 0.005
- Results - still too fast - accuracy bad: product .98
- Time is better 1.5 minutes vs. 3-3.5 minutes 
### Conclusions going forward:
- Go back to much lower learning rates - try 0.002 - twice as much as default

# Conclusions 19 - Learning rate 0.002 (twice the default 0.001)
- Results - still too fast - accuracy worse: product .995, but not .999
- Time took - slightly better than default learning rate 0.001
### Conclusions going forward:
- Go back to much lower learning rates - go back to 0.001

# Conclusions 20 - Large batch size (10,000)
- Much slower (700 sec instead of 300)
- Results not as good (accuracy product ~ .97-.985 instead of 99.5)
### Conclusions going forward:
- Try with batch size 400 (twice as much as current 200)

# Conclusions 21 - Larger batch size (400 instead of 200)
- Same or faster (200-350 sec instead of 300-400)
- Accuracies same or better (accuracy product ~ .9995+ instead of .9985-.9995)
### Conclusions going forward:
- Consider changing to this later, try 100 first

# Conclusions 22 - Smaller batch size (100 instead of 200)
- Faster (100-200 sec instead of 300-400)
- Accuracies worse (accuracy product ~ .995+ instead of .9985-.9995)
### Conclusions going forward:
- Change to 300 

# Conclusions 23 - Adding testing - 5 loops of the same
- Added testing - it gives lower results than expected - .97-.98, while validation and train accuracy is close to .9999
- Based on that, stopped updating the weights to best validity, and allowed to take last weights - simpler, and will allow to find best accuracy and patience
- Test accuracy: best .979, range .977-.979
- Product accuracy: best: .979, range .975-.979
- Validate and train accuracies: All .999 and up
- Train time - 205 seconds average, 170-230
- Num epochs 23-26 
### Conclusions going forward:
- Need another cycle with a lot of different parameters

# Conclusions 24 - Width 500 (instead 100)
- Test accuracy: best .981, range .977-.981 (same or better than 100)
- Product accuracy: best: .976, range .966-.976 (same or worse than 100)
- Validate and train accuracies: worse, .997-.998 (worse than 100)
- Train time - 145 seconds average (less than 205 of 100)
- Num epochs 13-16 - much less than 23-26 of 100
### Conclusions going forward:
- 500 doesn't seem to be better, but can't be ruled out altogether. Try with 200

# Conclusions 25 - Width 200 (instead 100)
- Test accuracy: best .983, range .977-.983 (probably better than 100)
- Product accuracy: best: .983, range .968-.983 (same or better than 100)
- Validate and train accuracies: worse, .997-.998 (worse than 100)
- Train time - 210 seconds average (a bit more than 100)
- Num epochs 20-26 - less than 23-26 of 100
### Conclusions going forward:
- Seems makes test better while train/validate worse. Leave 200

# Conclusions 26 - softmax 4 layers
- Goal: double check that softmax doesn't help with 4 layers
- Test Accuracy is some of the worst with softmax functions
- However, that could be because our StopFunction is still too strict
### Conclusions going forward:
- Remove for now softmax, work on making the StopFunction better, then try again with softmax

# Conclusions 27 - no limit stop function
- Goal: check how to improve stop function - let it run for longer
- val_accuracy keeps on being the same, while val_loss and loss are still improving
### Conclusions going forward:
- Start using val_loss as the stop function.  Need to play with patience.  Does it depend also on how fast we are learning? For example smaller batch size or larger learning rate = need to wait less to stop
- Try with var_loss that's any, and with patience 3

# Conclusions 28 - var_loss StopFuction with any var_loss, and with patience 3
- Patience 3 is too small - stopping too early
### Conclusions going forward:
- Try with patience 5

# Conclusions 29 - var_loss StopFunction with any var_loss, and with patience 5
- Patience 5 seems much better, but having no limit on loss causes loss to decrease for a very long time, but then just up again, and that's the final result we get
- Test function accuracye ~ .983
### Conclusions going forward:
- Limit delta to 0.00001

# Conclusions 30 - var_loss StopFunction with 0.00001, and with patience 5
- 2/3 stopped too early
- 1/3 stopped too late, when there was a jump to a bigger loss
### Conclusions going forward:
- Need more patience, but need to return to best result

# Conclusions 31 - var_loss 0.00001, patience 10, return to best results
- Product accuracy slightly better
- Test accuracy slightly better (2/3 similar .981, but 1 better - .983 instead of .974)
- Train time longer - 440 instead of 300
- Validate and train accuracies - all 1, while in patience 5 without returning .999-1
### Conclusions going forward:
- More expensive, but gives better results
- Introduce finding best relevant results


# Conclusions 32 - var_loss 0.0001, patience 7, return to best results
- Goal - Stop slightly earlier, not to take that much time
- Test accuracy worse than patience 10 (.98-.983)
- Train time - average 273 - shorter than 10 wiht 0.00001
- Validate and train accuracies - .995 and above - worse, not 1 as we got with 10 and 0.00001
### Conclusions going forward:
- Try 10 patience, but delta 0.0001 and not 0.00001

# Conclusions 33 - var_loss 0.0001, patience 10, return to best results
- Test accuracy .982 consistenly - similar to others before
- Train time - average 390 - saving 50 seconds from var_loss 0.00001
- Validate and train accuracies - mostly 1 (besides one case of .999)
- Validate loss - average .0002
- Train loss - average .0005
### Conclusions going forward:
- Staying with this configuration for now (var loss delta 0.0001 and patience 10)

# Conclusions 34 - learning rate higher 0.005 instead of default 0.001
- Much worse - not getting close to best result
### Conclusions going forward:
- Try with 0.0001

# Conclusions 35 - learning rate lower 0.0001 instead of default 0.001
- Test accuracy .9802 consistenly
- Train time 1516 - much higher
- Validate and train accuracy - 1
### Conclusions going forward:
- Accuracy is actually worse, so staying now with learning rate 0.001, but going forward to get better results, consider lowing learning rate from 0.001

# Conclusions 36 - different functions for 4 layers (all 4 activation functions, with everything else set to best so far)
- Activation functions
    - Softmax function clearly doesn't add anything - all results with it are bad
    - 1st sigmoid - not great results, although with not with softmax not much after other good ones
    - 1st relu/tanh - best, without major difference  of what the function is (but not softmax)
    - Absolutely best in every category: (not by much): ('relu', 'tanh'):
        - test accuracy .9828
        - test loss 0.0784
        - time 200 on a very fast computer (like 380 on mine)
### Conclusions going forward:
- Not include softmax in future tests, at least for 4 layers.  Sigmoid is probably not critical, so can leave just relu and tanh, but can definitely help in 5 layers
- Run different parameters only with 'relu' and 'tanh' on 4 layers

# Conclusions 37 - different functions for 5 layers (all 4 activation functions, with everything else set to best so far)
- Activation functions:
    - Softmax - doesn't seem to add much, usually worse, one time where it's good, usually having a different function is same or better
    - Sigmoid - gives same or better, when taken with relu and tanh together 
- Test accuracy
    - Best - .9842 test accuracy - ('relu', 'relu', 'tanh') with test loss .0812
    - A lot of the rest of top 10% are combinations of all 3 functions or 2/3, or even only relu
- Test loss
    - Best - .0715 test loss (with accuracy .9836) - ('tanh', 'tanh', 'sigmoid')
- Loss * time
    - A few of the best:
        - 2 above - ('relu', 'relu', 'tanh') with 17.4, and ('tanh', 'tanh', 'sigmoid') with 16.9
        - ('tanh', 'relu', 'tanh') one of the best in all categories - test accuracy 0.9833, test loss 0.08, loss * time = 16.9
- Best 4 functions:
    Hidden funcs	            Test Accuracy	Test Loss	Loss * Time
    ('tanh', 'tanh', 'sigmoid')	0.9836  	    0.0715	    16.9334
    ('tanh', 'relu', 'tanh')	0.9833  	    0.08	    16.9342
    ('relu', 'tanh', 'sigmoid')	0.9833  	    0.0808	    18.627
    ('relu', 'relu', 'tanh')	0.9842  	    0.0812	    17.4403
### Conclusions going forward:
- Play around with configuration  ('tanh', 'tanh', 'sigmoid'), perhaps with smaller learning rate, since seems train and validation loss were large, perhaps with smaller learning rate will do better
- Invest in different parameters for main combinations of tanh and relu  

# Conclusions 38 - learning rate 0.0005 - see if reaching better results
- Test accuracy - around .981 - not as good as 0.001
### Conclusions going forward:
- Stay with learning rate of 0.001

# Conclusions 39 - 5 layer - 'tanh', 'tanh', 'sigmoid' with a lot of patience and delta 0.00001 instead of 0.0001
- Test accuracy consistently 0.983
- Test loss 0.095 on average
- Train time 680 on average
### Conclusions going forward:
- Giving even more patience doesn't necessarily help - got .9836 in previous results
- Compare to doing the same with 4 layer 'tanh', 'relu'

# Conclusions 40 - 4 layer - 'tanh', 'relu', 'sigmoid' with a lot of patience and delta 0.00001 instead of 0.0001
- Test accuracy consistently 0.982
- Test loss 0.096 on average
- Train time 710 on average
### Conclusions going forward:
- Giving even more patience doesn't necessarily help - got same in previous results
- With given parameters this is the best option for 4 layers

# Conclusions 41 - 4 layers with different batch sizes, hidden widths and activation functions
- Test accuracy:
    - Best 0.9847 Batch size	Hidden funcs	    Hidden width
                    200	        ('relu', 'relu')	200
    - Top 10% - relu always first
    - Top 10% - Batch sizes vary
    - Top 10% - Hidden width - 300 almost everywhere
    - Top 10%:
        Batch size	Hidden funcs	Hidden width	Test Accuracy
        200	        ('relu', 'relu')	200	        0.9847
        200	        ('relu', 'relu')	300	        0.9843
        200	        ('relu', 'sigmoid')	300	        0.9841
        300	        ('relu', 'sigmoid')	300	        0.9839
        400	        ('relu', 'tanh')	300	        0.9839
- Test loss
    - Best: 0.0708 (average of best accuracies is .08)
        Batch size	Hidden funcs	    Hidden width	Test Accuracy	Test Loss
        300	        ('tanh', 'tanh')	300	            0.9831          0.0708
    - Batch size and hidden width vary
### Conclusions going forward:
- Try higher hidden width than 300
- Try around following values (accuracy .9847): 
  Batch size	Hidden funcs	    Hidden width
  200	        ('relu', 'relu')	200

# Conclusions 42 - 4 layer - 'relu', 'relu' widths and batches around 200, 200
- relu relu with 200 200 is the best option - test accuracy 0.9840 (rest are a bit less)
### Conclusions going forward:
- So far for 4 layers relu relu with 200 200 is the best option (accuracy .9840)
- Wait for results of 4 layers with higher width than 300
- Wait for results of 5 layers with different parameters

# Conclusions 43 - 5 layer - (tanh, relu, relu) widths and batches  300, 300
- Goal: Saw that it's one of the best options in 5 layers, so tried it locally
- Test accuracy: .982-.983 - not as good as what I have with 4 layers
### Conclusions going forward:
- So far the option with 4 layers is best

# Conclusions 44 - 4 layers - different func, and batches (200,300,400), but higher width 400
- relu relu slightly better with width 400 and batch 400 (.985 test accuracy)
### Conclusions going forward:
- Up relu relu width and batch to 400, try higher on both

# Conclusions 45 - 5 layers - different func, and batches (200,300,400), and widths (200, 300)
- Best test accuracy - top 5%
    - Batch sizes vary 
    - Widths - mostly 300, the ones with 200, usually 300 is similar
    - Best: 300	('tanh', 'relu', 'relu')	300 with test accuracy of 0.9862 on AWS, however locally got only .982/.983
    - 2nd best: 300	('relu', 'tanh', 'sigmoid')	300	accuracy 0.9857 
### Conclusions going forward:
- Consider width > 300
- Consider 2nd best above

# Conclusions 46 - 5 layers - ('relu', 'tanh', 'sigmoid') with 300,300 batch and width
- 3 tries gave very different results of test accuracy - .9823 - .9854
### Conclusions going forward:
- Can't consistently use this over 4 layers that gives more consistently .984


# Conclusions 47 - 4 layers - ('relu', 'relu') with batch [350,400,450] and widths [350,400,450]
- Best test accuracy: 400 batch, 450 width: test accuracy 0.9848
- 2nd best: 350 batch, 400 width: test accuracy 0.9845
- 3rd best: 450 batch, 450 width
### Conclusions going forward:
- Try batch 400, 450 with width 500


# Conclusions 48 - 4 layers - ('relu', 'relu') with batch 400 and widths 450
- Local test (not in AWS) - not exactly the same results in both
- Accuracy: .983-.984, 2/3 .984
### Conclusions going forward:
- Compare to tanh, relu local 200, 200

# Conclusions 49 - 4 layers - ('relu', 'relu') with batch and widths around 450
- On AWS:
- Both 450 seems the best - test accuracy .9848. Both some are good with 400, 450, 500.  So staying in the middle  
### Conclusions going forward:
- Change batch and width to 450

# Conclusions 50 - 4 layers - ('tanh', 'relu') with batch and widths 200
- Local test (not in AWS)
- Goal: compare to relu relu local
- Accuracy - on average .983 (.982-.984)
### Conclusions going forward:
- relu relu seems better

# Conclusions 51 - 5 layers - width 400, different activation functions and batches (200,300,400)
- AWS run
- Top 5%:
    Hidden width	Batch size	Hidden funcs	            Test Accuracy
    400	            200	        ('relu', 'tanh', 'sigmoid')	0.9852
    400	            300	        ('relu', 'relu', 'relu')	0.9851
    400	            300	        ('tanh', 'relu', 'sigmoid')	0.9856
    400	            400	        ('tanh', 'relu', 'tanh')	0.9859
- Best: 400	            400	        ('tanh', 'relu', 'tanh')	0.9859, see next run locally
- Second - ('tanh', 'relu', 'sigmoid') that was tried and didn't give consistent better results locally
- sigmoid as 3rd with relu,tanh or tanh,relu gives great results, but doesn't seem extremely consistent
### Conclusions going forward:
- Test locally the best run, compare to 4 layers best solution

# Conclusions 52 - 5 layers - (tanh, relu, tanh), batch, width 450, 450
- Local run
- Average test accuracy of .985, somewhat consistent. Possibly slightly better by .001 than 4 layer solution.
- Time: 410 local
- Test loss average: 0.675  
### Conclusions going forward:
- Not worth going to a 5 layer slower less stable solution for .001. Staying with 4 layers 

# Conclusions 53 - 4 layers ('relu', 'relu'), batch, width 450, 450
- Local run
- Goal compare to 5 layers above 
- Average test accuracy: .9835 (.983-.984)
- Time: 345 local
- Test loss: 0.08
### Conclusions going forward:
- 5 layers model above seems to be slightly better, slower and less consistent.  Staying with 4 layers.

# Conclusions 54 - 4 layers ('relu', 'relu'), batch 450, width 784 (number of inputs)
- Local run
- Goal - try to have width at least as number of inputs (actually exactly in this case) 
- Average test accuracy: .9843 (.983-.984)
- Time: 485 average local
- Test loss: 0.08
### Conclusions going forward:
- Doesn't seem to improve by much from 450 width to 784.  Leave 450, but as an option for running different options, leave in

# Conclusions 55 - 5 layers (tanh, relu, tanh), batch 450, width 784 (number of inputs)
- Local run
- Goal - try to have width at least as number of inputs (actually exactly in this case) 
- Average test accuracy: .982 - worse than 450
- Time: 450 average local
- Test loss: 0.08
### Conclusions going forward:
- Doesn't seem to improve by much from 450 width to 784.  Leave 450, but as an option for running different options, leave in
