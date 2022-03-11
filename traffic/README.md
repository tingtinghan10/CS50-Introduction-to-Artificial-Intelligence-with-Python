Started with the sample code from lecture, and experimented from there.
The initial accuracy was low at only around 5%.
Added a convolution and hidden layer which didn't help with accuracy.
Increased pool size of the second hidden layer.
The process took approximately the same time but accuracy significantly improved.
Added a third convolution layer, accuracy improved but still low. ***
Tested dropout at 0.4, 0.5 and 0.6 and a dropout of 0.5 gave the highest accuracy.
The final model gives and accuracy of over 95%.