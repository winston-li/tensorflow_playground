These python files exercised TensorFlow model's serialization and deserialization, so that 
we can do continuous training instead of one-shot only. In addition to the training & validation
process on powerful devices (or in the cloud), it also exercised finalize a model, so that it 
could be deployed to embedded devices for run-time prediction.

### Steps
- Training machines:

    (1) Training:
       
    Repeatly execute it for continuous training

    ```
    $ nohup python mnist_train.py &
    ```
    (2) Evaluation:
    ```    
    $ python mnist_eval.py
        
      Validation accuracy:
      Restore from  /home/Winston/python/mnist/models/model.ckpt-11000
      Accuracy: 0.97620, elipsed time: 3.78499 sec for 5000 samples
      Test accurary:
      Restore from  /home/Winston/python/mnist/models/model.ckpt-11000
      Accuracy: 0.97680, elipsed time: 7.47098 sec for 10000 samples
    ```
    (3) Finalize Model:
    ```
    $ python mnist_freeze_model.py 
    
      Restore from  /home/Winston/python/mnist/models/model.ckpt-11000
      Merge GraphDef and Variables from 
        /home/Winston/python/mnist/models/draft_model.pb 
        /home/Winston/python/mnist/models/model.ckpt-11000
      Converted 8 variables to const ops.
      48 ops in the final graph.
      Finalized Graph /home/Winston/python/mnist/models/model.pb written
    ```

- Runtime devices:
    Copy the finalized model to devices.

    (4) Predict:
    ```
   $ python mnist_predict.py 
    
     Round 0: predicts = [7 2 1 0 4 1 4 9 5 9] 
     Round 0: labels   = [7 2 1 0 4 1 4 9 5 9] 
     Round 1: predicts = [0 6 9 0 1 5 9 7 8 4] 
     Round 1: labels   = [0 6 9 0 1 5 9 7 3 4] 
     Round 2: predicts = [9 6 6 5 4 0 7 4 0 1] 
     Round 2: labels   = [9 6 6 5 4 0 7 4 0 1]  
    ```

### Notes
- Placeholders are used for images, labels, and keep_rate (dropout layer), so that
  it's easier to control the data feed/flow.
- Average loss and Expoential Moving Average loss across mini-batches are recorded instead of just the most
  recent mini-batch's loss.

