Based on mnist model/training/evaluation/predict skeleton, this subproject
exercises another model & dataset for drone autonomous flight via visual perception.
The model & dataset refer to http://people.idsia.ch/~guzzi/DataSet.html

### Steps
- Training machines:

    (1) Training:
       
    Repeatly execute it for continuous training

    ```
    $ nohup python drone_train.py &
    ```
    (2) Evaluation:
    ```    
    $ python drone_eval.py
    ```
    (3) Finalize Model:
    ```
    $ python drone_freeze_model.py 

    ```

- Runtime devices:
    Copy the finalized model to devices.

    (4) Predict:
    ```
   $ python drone_predict.py 
    
    ```

### Notes
- Placeholders are used for images, labels, and keep_rate (dropout layer), so that
  it's easier to control the data feed/flow.
- Average loss and Expoential Moving Average loss across mini-batches are recorded instead of just the most
  recent mini-batch's loss.

