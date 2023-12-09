# Lab2 Scalable ML

## Improving Performance
### Model Centric Approach
The model centric aproach of improving performance involves changing the architechture of the model and the hyperparameters.
One can in many cases obtain better performance by increasing the capacity of the model for example by increasin its size.
It's clear that the larger whisper models are used in the best performing results for speech recognition. However larger models
often require more data to prevent overfitting and they also are more resource intensive. In this lab, the larger feasible model 
size to train on was was the small whisper model. We instead opted to change the hyperparameters in order to get better performnce. 
The base implementation used a constant learning rate with warmup. We were able to obtain better performance by lowering the learning
rate at the later epochs. This way the model is able to make finer steps at the end when it has reach better solutions in other to 
make final finetuning adjustments. We therefore kept the warmup by changed the schedule to a cosine schedule which deacreased to zero over time. 
We also increased the number of steps from 4000 to 6000 in order to give the modle more time to ave relatively high learning rate and reach
a decent solution before making the final adjustments.

### Data Centric Approach
Another approach is to make changes to the dataset. in general models with more training data are able to learn a better solution with better 
performance metrics and generalization capabilities, assuming it has eonugh capacity. The datasize can be increased simply by adding more samples
it can also be effectively made larger through different augmentations. We tried to increase the size of the dataset by combining it with another dataset.
This was achieved by combining the base, daataset common voice 13 with the google felurs dataset. However, we did not see any improvement in performance.
We also tried to apply some augmentations to the data. We did this by randomly masking the time and frequency values of the mel spectrogram for the data[1].
We found that this did lead to better performance.

[1] https://arxiv.org/abs/1904.08779
