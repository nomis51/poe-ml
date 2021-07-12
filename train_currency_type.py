from train import train_until_threshold, fix_crash

size = (46, 46)
batch_size = 16
training_dir = "./images/currency_types/training/"
test_dir = training_dir
training_name = "currency_type"
new_training = True
epochs = 30
step_epoch = 10
expected_accuracy = 0.99

fix_crash()
train_until_threshold(training_name, training_dir, size, test_dir,
                      expected_accuracy, batch_size, epochs, step_epoch, new_training)
