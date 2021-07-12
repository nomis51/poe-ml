from train import train_until_threshold, fix_crash

size = (46, 46)
batch_size = 16
epochs = 100
training_dir = "./images/stack_sizes/training/"
test_dir = training_dir
training_name = "stack_size"
new_training = True
epochs = 2700
step_epoch = 200
expected_accuracy = 1.0

fix_crash()
train_until_threshold(training_name, training_dir, size, test_dir,
                      expected_accuracy, batch_size, epochs, step_epoch, new_training)
